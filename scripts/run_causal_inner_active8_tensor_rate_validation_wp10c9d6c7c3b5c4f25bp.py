#!/usr/bin/env python3
"""Fit and independently validate the active-8 full-tensor architecture."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

import run_causal_inner_active8_tensor_database_extension_manifest_wp10c9d6c7c3b5c4f25bn as manifest  # noqa: E402
import run_causal_inner_active8_tensor_geometry_extension_wp10c9d6c7c3b5c4f25bo as parent  # noqa: E402
import run_causal_inner_active8_tensor_architecture_diagnosis_wp10c9d6c7c3b5c4f25bm as architecture  # noqa: E402
import run_causal_inner_active8_mixed_parity_rate_fit_wp10c9d6c7c3b5c4f25bl as old_rate  # noqa: E402
import run_causal_inner_expanded_departure_rate_screen_wp10c9d6c7c3b5c4f25be as rate_tools  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bp"
GEOMETRY_COMMIT = "d470b89b3460394834067e2776e9aebce8cd45d3"
GEOMETRY_PARENT = "28ac5ef7c8e9fc8d08c268b62ede39bec6ec9c7c"
GEOMETRY_TREE = "81995f444169db7e55e2413ddbe14eb381067fc0"

PASS_CLASSIFICATION = (
    "active8_full_tensor_rate_and_rank4_slaved_curvature_"
    "independently_validated"
)
FAIL_CLASSIFICATION = (
    "active8_tensor_database_or_independent_validation_failed"
)
AUTHORIZED_NEXT = (
    "definitions_only_active8_short_reduced_vector_field_validation_manifest"
)

ARTIFACT = (
    "causal_inner_active8_tensor_rate_validation_"
    "wp10c9d6c7c3b5c4f25bp"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_tensor_rate_validation_"
    "wp10c9d6c7c3b5c4f25bp.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_tensor_rate_validation_"
    "wp10c9d6c7c3b5c4f25bp.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_TENSOR_RATE_"
    "VALIDATION_WP10C9D6C7C3B5C4F25BP_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

DATABASE_PATH = parent.CANONICAL_DIRECTORY / "tensor_geometry_database.npz"
DESIGN_PATH = manifest.CANONICAL_DIRECTORY / "extension_design.npz"
OLD_CLOSURE_PATH = old_rate.CANONICAL_DIRECTORY / "mixed_parity_closure.npz"
ONLINE_GEOMETRY_PATH = old_rate.manifest.GEOMETRY_PATH
GENERATOR_PATH = old_rate.GENERATOR_PATH
FIT_ARRAY_PATH = SCRATCH_DIRECTORY / "frozen_coefficients.npz"
FIT_LOCK_PATH = SCRATCH_DIRECTORY / "coefficient_lock.json"


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


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(expected))
        / max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    )


def _append(array: np.ndarray, value) -> np.ndarray:
    item = np.asarray(value, dtype=float)
    return np.concatenate((array, item.reshape((1,) + item.shape)), axis=0)


def _evaluation_order() -> tuple[int, ...]:
    training = tuple(range(0, manifest.NEW_TRAINING_CANDIDATES))
    tuning_high = tuple(
        range(
            manifest.NEW_TRAINING_CANDIDATES,
            manifest.NEW_TRAINING_CANDIDATES + 2 * manifest.NEW_TUNING_DIRECTIONS,
        )
    )
    holdout_start = (
        manifest.NEW_TRAINING_CANDIDATES + 2 * manifest.NEW_TUNING_DIRECTIONS
    )
    holdout = tuple(range(holdout_start, holdout_start + manifest.NEW_HOLDOUT_CANDIDATES))
    tuning_low = tuple(
        range(
            holdout_start + manifest.NEW_HOLDOUT_CANDIDATES,
            manifest.PLANNED_CANDIDATES,
        )
    )
    return training + tuning_high + tuning_low + holdout


def _validate_geometry(*, require_clean: bool) -> dict:
    if GEOMETRY_COMMIT.startswith("TO_BE_FILLED"):
        raise RuntimeError("tensor-geometry result lineage has not been frozen")
    if _git("rev-parse", GEOMETRY_COMMIT) != GEOMETRY_COMMIT:
        raise RuntimeError("tensor-geometry result commit changed")
    if _git("rev-parse", f"{GEOMETRY_COMMIT}^") != GEOMETRY_PARENT:
        raise RuntimeError("tensor-geometry result lineage changed")
    if _git("rev-parse", f"{GEOMETRY_COMMIT}^{{tree}}") != GEOMETRY_TREE:
        raise RuntimeError("tensor-geometry result tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["completed_candidate_count"] != manifest.PLANNED_CANDIDATES
        or summary["failed_candidate_count"] != 0
        or summary["nonbase_continuous_rate_evaluations"] != 0
        or not all(metrics["checks"].values())
    ):
        raise RuntimeError("tensor-geometry rate authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"geometry source changed: {relative}")
    _checksums(manifest.CANONICAL_DIRECTORY)
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("tensor rate validation requires a clean tracked tree")
    for name, expected in parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _load_inputs() -> dict:
    database = _load_npz(DATABASE_PATH)
    online_geometry = _load_npz(ONLINE_GEOMETRY_PATH)
    generator = _load_npz(GENERATOR_PATH)
    old_closure = _load_npz(OLD_CLOSURE_PATH)
    design = _load_npz(DESIGN_PATH)
    metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    states = np.asarray(database["candidate_primitive_states"], dtype=float)
    deltas = np.asarray(database["candidate_scaled_deltas"], dtype=float)
    coordinates = np.asarray(
        database["candidate_departure_coordinates"], dtype=float
    )
    candidates = metrics["candidates"]
    if (
        states.shape != (manifest.PLANNED_CANDIDATES, 112, 5)
        or deltas.shape != (manifest.PLANNED_CANDIDATES, 560)
        or coordinates.shape != (manifest.PLANNED_CANDIDATES, 28)
        or len(candidates) != manifest.PLANNED_CANDIDATES
        or generator["complete_fixed_Q_generator"].shape != (560, 560)
        or online_geometry["online_coordinate_restriction"].shape != (470, 560)
        or design["rank4_curvature_basis"].shape != (560, 4)
        or old_closure["candidate_scaled_deltas"].shape[0] < 112
    ):
        raise RuntimeError("tensor rate input dimensions changed")
    if [item["candidate_index"] for item in candidates] != list(
        range(manifest.PLANNED_CANDIDATES)
    ):
        raise RuntimeError("tensor geometry candidate ordering changed")
    return {
        "database": database,
        "online_geometry": online_geometry,
        "generator": np.asarray(
            generator["complete_fixed_Q_generator"], dtype=float
        ),
        "base_rate": np.asarray(generator["fixed_Q_rate"], dtype=float),
        "old_closure": old_closure,
        "curvature_basis": np.asarray(
            design["rank4_curvature_basis"], dtype=float
        ),
        "states": states,
        "deltas": deltas,
        "coordinates": coordinates,
        "candidates": candidates,
    }


def _progress_array_names() -> tuple[str, ...]:
    return (
        "total_rates_per_second",
        "free_rates_per_second",
        "physical_reaction_actions_per_second",
        "multiplier_coordinates_per_second",
        "online_470_coordinate_rates_per_second",
        "departure_rate_increments_per_second",
        "linear_rate_references_per_second",
        "departure_linear_references_per_second",
    )


def _progress_identity() -> dict:
    return {
        "execution_commit": _git("rev-parse", "HEAD"),
        "geometry_commit": GEOMETRY_COMMIT,
        "geometry_database_sha256": _sha(DATABASE_PATH),
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
        raise RuntimeError("tensor rate scratch checkpoint is incomplete")
    recorded = _read(json_path)
    if recorded["identity"] != identity:
        raise RuntimeError("tensor rate scratch identity changed")
    progress = {
        "identity": identity,
        "evaluations": recorded["evaluations"],
        "failures": recorded["failures"],
        **_load_npz(npz_path),
    }
    count = len(progress["evaluations"])
    if any(progress[name].shape[0] != count for name in _progress_array_names()):
        raise RuntimeError("tensor rate scratch dimensions changed")
    expected_prefix = list(_evaluation_order()[:count])
    actual = [item["candidate_index"] for item in progress["evaluations"]]
    if actual != expected_prefix:
        raise RuntimeError("tensor rate evaluation order changed")
    return progress


def _evaluate_candidate(
    inputs: dict, progress: dict, index: int, data: dict, components: dict
) -> None:
    state = inputs["states"][index]
    candidate = inputs["candidates"][index]
    item, arrays = rate_tools.manifest.prior_screen._continuous_rate(data, state)
    coordinate_jacobian, coordinate_metrics = parent.chart_tools._coordinate_jacobian(
        state, components
    )
    linear = inputs["generator"] @ inputs["deltas"][index]
    increment = arrays["total_rate"] - inputs["base_rate"]
    departure_increment = (
        inputs["online_geometry"]["departure_coordinate_basis"].T @ increment
    )
    departure_linear = (
        inputs["online_geometry"]["departure_coordinate_basis"].T @ linear
    )
    online_rate = np.concatenate(
        (
            coordinate_jacobian @ arrays["total_rate"],
            inputs["online_geometry"]["stable_memory_coordinate_basis"].T
            @ arrays["total_rate"],
            inputs["online_geometry"]["departure_coordinate_basis"].T
            @ arrays["total_rate"],
        )
    )
    item.update(
        {
            "candidate_index": index,
            "pair_index": candidate["pair_index"],
            "split": candidate["split"],
            "split_direction_index": candidate["split_direction_index"],
            "global_direction_index": candidate["global_direction_index"],
            "component_bound": candidate["component_bound"],
            "amplitude_label": candidate["amplitude_label"],
            "sign": candidate["sign"],
            "state_rate_linear_relative_defect": _relative_error(increment, linear),
            "departure_rate_linear_relative_defect": _relative_error(
                departure_increment, departure_linear
            ),
            "coordinate_Jacobian_rank": coordinate_metrics["rank"],
            "coordinate_Jacobian_condition_number": coordinate_metrics[
                "condition_number"
            ],
        }
    )
    progress["evaluations"].append(item)
    for name, value in (
        ("total_rates_per_second", arrays["total_rate"]),
        ("free_rates_per_second", arrays["free_rate"]),
        ("physical_reaction_actions_per_second", arrays["reaction_action"]),
        ("multiplier_coordinates_per_second", arrays["multiplier"]),
        ("online_470_coordinate_rates_per_second", online_rate),
        ("departure_rate_increments_per_second", departure_increment),
        ("linear_rate_references_per_second", linear),
        ("departure_linear_references_per_second", departure_linear),
    ):
        progress[name] = _append(progress[name], value)


def _pair_targets(
    *,
    deltas: np.ndarray,
    coordinates: np.ndarray,
    departure_increments: np.ndarray,
    departure_linear: np.ndarray,
    energy_directions: np.ndarray,
    curvature_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    if deltas.shape[0] % 2:
        raise RuntimeError("signed training data are not paired")
    nonlinear = departure_increments - departure_linear
    directions = []
    radii = []
    quadratic = []
    cubic = []
    curvature_cubic = []
    for negative in range(0, deltas.shape[0], 2):
        positive = negative + 1
        active_negative = energy_directions.T @ coordinates[negative]
        active_positive = energy_directions.T @ coordinates[positive]
        active_odd = 0.5 * (active_positive - active_negative)
        radius = float(np.linalg.norm(active_odd))
        if radius <= np.finfo(float).tiny:
            raise RuntimeError("active training radius vanished")
        direction = active_odd / radius
        curvature_negative = deltas[negative] @ curvature_basis
        curvature_positive = deltas[positive] @ curvature_basis
        directions.append(direction)
        radii.append(radius)
        quadratic.append(
            0.5 * (nonlinear[positive] + nonlinear[negative]) / radius**2
        )
        cubic.append(
            0.5 * (nonlinear[positive] - nonlinear[negative]) / radius**3
        )
        curvature_cubic.append(
            0.5 * (curvature_positive - curvature_negative) / radius**3
        )
    return {
        "directions": np.asarray(directions, dtype=float),
        "radii": np.asarray(radii, dtype=float),
        "rate_quadratic_targets": np.asarray(quadratic, dtype=float),
        "rate_cubic_targets": np.asarray(cubic, dtype=float),
        "curvature_cubic_targets": np.asarray(curvature_cubic, dtype=float),
    }


def _training_targets(inputs: dict, progress: dict) -> dict[str, np.ndarray]:
    training_count = manifest.NEW_TRAINING_CANDIDATES
    if len(progress["evaluations"]) < training_count or [
        item["candidate_index"]
        for item in progress["evaluations"][:training_count]
    ] != list(range(training_count)):
        raise RuntimeError("coefficient fit attempted before training truth completed")
    old = inputs["old_closure"]
    energy = np.asarray(inputs["database"]["energy_directions"], dtype=float)
    old_targets = _pair_targets(
        deltas=old["candidate_scaled_deltas"][:112],
        coordinates=old["candidate_departure_coordinates"][:112],
        departure_increments=old["departure_rate_increments_per_second"][:112],
        departure_linear=old["departure_linear_references_per_second"][:112],
        energy_directions=energy,
        curvature_basis=inputs["curvature_basis"],
    )
    new_targets = _pair_targets(
        deltas=inputs["deltas"][: manifest.NEW_TRAINING_CANDIDATES],
        coordinates=inputs["coordinates"][: manifest.NEW_TRAINING_CANDIDATES],
        departure_increments=progress["departure_rate_increments_per_second"][
            :training_count
        ],
        departure_linear=progress["departure_linear_references_per_second"][
            :training_count
        ],
        energy_directions=energy,
        curvature_basis=inputs["curvature_basis"],
    )
    combined = {
        name: np.concatenate((old_targets[name], new_targets[name]), axis=0)
        for name in old_targets
    }
    if combined["directions"].shape != (manifest.TOTAL_TRAINING_DIRECTIONS, 8):
        raise RuntimeError("combined tensor-training direction count changed")
    return combined


def _fit_coefficients(targets: dict[str, np.ndarray]) -> tuple[dict, dict]:
    directions = targets["directions"]
    quadratic = architecture.parent.manifest._quadratic_features(directions)
    cubic = architecture._cubic_features(directions)
    metrics = {
        "training_direction_count": int(directions.shape[0]),
        "actual_quadratic_feature_rank": int(np.linalg.matrix_rank(quadratic)),
        "actual_quadratic_feature_condition_number": float(np.linalg.cond(quadratic)),
        "actual_cubic_feature_rank": int(np.linalg.matrix_rank(cubic)),
        "actual_cubic_feature_condition_number": float(np.linalg.cond(cubic)),
    }
    if metrics["actual_quadratic_feature_rank"] != 36:
        raise RuntimeError("actual quadratic training design lost rank")
    if metrics["actual_cubic_feature_rank"] != 120:
        raise RuntimeError("actual cubic training design lost rank")
    coefficients = {
        "rate_quadratic_coefficients": np.linalg.lstsq(
            quadratic, targets["rate_quadratic_targets"], rcond=None
        )[0],
        "rate_cubic_coefficients": np.linalg.solve(
            cubic, targets["rate_cubic_targets"]
        ),
        "curvature_cubic_coefficients": np.linalg.solve(
            cubic, targets["curvature_cubic_targets"]
        ),
    }
    return metrics, {**targets, **coefficients}


def _freeze_or_validate_coefficients(inputs: dict, progress: dict) -> tuple[dict, dict]:
    if len(progress["evaluations"]) < manifest.NEW_TRAINING_CANDIDATES:
        raise RuntimeError("cannot freeze coefficients before training completion")
    if FIT_LOCK_PATH.exists() != FIT_ARRAY_PATH.exists():
        raise RuntimeError("coefficient-lock checkpoint is incomplete")
    if FIT_LOCK_PATH.exists():
        lock = _read(FIT_LOCK_PATH)
        if (
            lock["coefficient_sha256"] != _sha(FIT_ARRAY_PATH)
            or lock["training_candidate_count"] != manifest.NEW_TRAINING_CANDIDATES
            or lock["validation_rate_evaluations_at_freeze"] != 0
            or lock["runner_sha256"] != _sha(ROOT / THIS_RUNNER)
            or lock["geometry_database_sha256"] != _sha(DATABASE_PATH)
            or lock["old_closure_sha256"] != _sha(OLD_CLOSURE_PATH)
        ):
            raise RuntimeError("frozen coefficient lock changed")
        arrays = _load_npz(FIT_ARRAY_PATH)
        current_targets = _training_targets(inputs, progress)
        for name, values in current_targets.items():
            if name not in arrays or not np.array_equal(values, arrays[name]):
                raise RuntimeError("training truth changed after coefficient freeze")
        return lock["fit_metrics"], arrays
    if len(progress["evaluations"]) != manifest.NEW_TRAINING_CANDIDATES:
        raise RuntimeError("validation truth was seen before coefficient freeze")
    targets = _training_targets(inputs, progress)
    fit_metrics, arrays = _fit_coefficients(targets)
    _write_npz(FIT_ARRAY_PATH, arrays)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "training_candidate_count": manifest.NEW_TRAINING_CANDIDATES,
        "total_training_direction_count": manifest.TOTAL_TRAINING_DIRECTIONS,
        "validation_rate_evaluations_at_freeze": 0,
        "coefficient_sha256": _sha(FIT_ARRAY_PATH),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "geometry_database_sha256": _sha(DATABASE_PATH),
        "old_closure_sha256": _sha(OLD_CLOSURE_PATH),
        "fit_metrics": fit_metrics,
    }
    _write_json(FIT_LOCK_PATH, lock)
    print(
        json.dumps(
            {
                "coefficient_lock_frozen": True,
                "training_candidates": manifest.NEW_TRAINING_CANDIDATES,
                "coefficient_sha256": lock["coefficient_sha256"],
            }
        ),
        flush=True,
    )
    return fit_metrics, arrays


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
                raise RuntimeError("cannot order incomplete truth arrays")
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
        "minimum_reconstruction_factor": minimum(
            "minimum_reconstruction_factor", math.inf
        ),
        "maximum_reconstruction_factor": maximum("maximum_reconstruction_factor"),
        "maximum_raw_Schur_condition_number": maximum(
            "raw_Schur_condition_number"
        ),
        "maximum_reaction_identity_defect": maximum("reaction_identity_defect"),
        "maximum_rate_tangency_relative_defect": maximum(
            "rate_tangency_relative_defect"
        ),
        "maximum_coordinate_Jacobian_condition_number": maximum(
            "coordinate_Jacobian_condition_number"
        ),
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum(
            "minimum_scattering_optical_depth"
        ),
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


def _execute_truth(inputs: dict) -> tuple[dict, dict[str, np.ndarray], dict, dict]:
    progress = _load_or_create_progress()
    resumed = len(progress["evaluations"])
    began = time.perf_counter()
    data = rate_tools.manifest.prior_screen.manifest.parent.manifest.failed_screen._anchor_data(
        "primary"
    )
    components = parent.high_chart._prepare_components()
    fit_metrics = {}
    fit_arrays = {}
    order = _evaluation_order()
    if len(progress["evaluations"]) >= manifest.NEW_TRAINING_CANDIDATES:
        fit_metrics, fit_arrays = _freeze_or_validate_coefficients(inputs, progress)
    for position in range(len(progress["evaluations"]), len(order)):
        if position == manifest.NEW_TRAINING_CANDIDATES:
            fit_metrics, fit_arrays = _freeze_or_validate_coefficients(
                inputs, progress
            )
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
    if (
        len(progress["evaluations"]) >= manifest.NEW_TRAINING_CANDIDATES
        and not progress["failures"]
        and not fit_arrays
    ):
        fit_metrics, fit_arrays = _freeze_or_validate_coefficients(inputs, progress)
    metrics, arrays = _truth_metrics(progress, began)
    metrics["resumed_evaluation_count"] = resumed
    metrics["coefficient_lock_frozen"] = FIT_LOCK_PATH.exists()
    return metrics, arrays, fit_metrics, fit_arrays


def _truth_gate_checks(metrics: dict, gates: dict) -> dict:
    return {
        "completed": metrics["completed_nonbase_rate_evaluations"]
        == gates["completed_nonbase_rate_evaluations_equal"],
        "failed": metrics["failed_rate_evaluations"]
        == gates["failed_rate_evaluations_equal"],
        "reconstruction_minimum": metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"],
        "reconstruction_maximum": metrics["maximum_reconstruction_factor"]
        <= gates["maximum_reconstruction_factor"],
        "Schur_condition": metrics["maximum_raw_Schur_condition_number"]
        <= gates["maximum_raw_Schur_condition_number"],
        "reaction_identity": metrics["maximum_reaction_identity_defect"]
        <= gates["maximum_reaction_identity_defect"],
        "rate_tangency": metrics["maximum_rate_tangency_relative_defect"]
        <= gates["maximum_rate_tangency_relative_defect"],
        "coordinate_condition": metrics[
            "maximum_coordinate_Jacobian_condition_number"
        ]
        <= gates["maximum_coordinate_Jacobian_condition_number"],
        "height": metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"],
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"],
        "incoming_excision": metrics["maximum_incoming_excision_characteristics"]
        == gates["maximum_incoming_excision_characteristics_equal"],
        "generator_budget": metrics["new_complete_generator_assemblies"]
        == gates["new_complete_generator_assemblies_equal"],
        "root_budget": metrics["new_nonlinear_roots"]
        == gates["new_nonlinear_roots_equal"],
        "propagation_budget": metrics["propagated_states"]
        == gates["propagated_states_equal"],
        "coefficient_lock": metrics["coefficient_lock_frozen"],
    }


def _predict(active: np.ndarray, coefficients: dict) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(active, dtype=float).reshape(1, 8)
    quadratic = architecture.parent.manifest._quadratic_features(point)[0]
    cubic = architecture._cubic_features(point)[0]
    nonlinear = (
        quadratic @ coefficients["rate_quadratic_coefficients"]
        + cubic @ coefficients["rate_cubic_coefficients"]
    )
    curvature = cubic @ coefficients["curvature_cubic_coefficients"]
    return nonlinear, curvature


def _validation_metrics(
    inputs: dict, truth: dict[str, np.ndarray], coefficients: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    energy = np.asarray(inputs["database"]["energy_directions"], dtype=float)
    restriction = inputs["online_geometry"]["online_coordinate_restriction"]
    lifting = inputs["online_geometry"]["online_coordinate_lifting"]
    components = parent.high_chart._prepare_components()
    nonlinear_truth = (
        truth["departure_rate_increments_per_second"]
        - truth["departure_linear_references_per_second"]
    )
    records = []
    predicted_nonlinear = np.full((manifest.PLANNED_CANDIDATES, 28), np.nan)
    predicted_curvature = np.full((manifest.PLANNED_CANDIDATES, 4), np.nan)
    predicted_deltas = np.full((manifest.PLANNED_CANDIDATES, 560), np.nan)
    for index, candidate in enumerate(inputs["candidates"]):
        if candidate["split"] == "training":
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
        coordinate, coordinate_factors = parent.chart_tools._coordinate_value_with_factors(
            state, components
        )
        physical = parent.chart_tools._state_audit(components["context"], state)
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

    tuning = [item for item in records if item["split"].startswith("tuning")]
    holdout = [item for item in records if item["split"] == "holdout"]
    combined = tuning + holdout

    def aggregate(items: list[dict], field: str, operation) -> float:
        return float(operation([item[field] for item in items]))

    metrics = {
        "tuning_candidate_count": len(tuning),
        "holdout_candidate_count": len(holdout),
        "tuning_median_nonlinear_departure_rate_relative_error": aggregate(
            tuning, "nonlinear_departure_rate_relative_error", np.median
        ),
        "tuning_maximum_nonlinear_departure_rate_relative_error": aggregate(
            tuning, "nonlinear_departure_rate_relative_error", np.max
        ),
        "holdout_median_nonlinear_departure_rate_relative_error": aggregate(
            holdout, "nonlinear_departure_rate_relative_error", np.median
        ),
        "holdout_maximum_nonlinear_departure_rate_relative_error": aggregate(
            holdout, "nonlinear_departure_rate_relative_error", np.max
        ),
        "tuning_median_full_departure_rate_relative_error": aggregate(
            tuning, "full_departure_rate_relative_error", np.median
        ),
        "tuning_maximum_full_departure_rate_relative_error": aggregate(
            tuning, "full_departure_rate_relative_error", np.max
        ),
        "holdout_median_full_departure_rate_relative_error": aggregate(
            holdout, "full_departure_rate_relative_error", np.median
        ),
        "holdout_maximum_full_departure_rate_relative_error": aggregate(
            holdout, "full_departure_rate_relative_error", np.max
        ),
        "maximum_curvature_prediction_error_over_full_state_delta": aggregate(
            combined, "curvature_prediction_error_over_full_state_delta", np.max
        ),
        "maximum_full_scaled_state_decoder_relative_error": aggregate(
            combined, "full_scaled_state_decoder_relative_error", np.max
        ),
        "maximum_reconstructed_C_phys_residual_infinity": aggregate(
            combined, "reconstructed_C_phys_residual_infinity", np.max
        ),
        "minimum_reconstructed_state_reconstruction_factor": aggregate(
            combined, "minimum_reconstructed_state_reconstruction_factor", np.min
        ),
        "maximum_reconstructed_H_over_R": aggregate(
            combined, "maximum_reconstructed_H_over_R", np.max
        ),
        "minimum_reconstructed_scattering_optical_depth": aggregate(
            combined, "minimum_reconstructed_scattering_optical_depth", np.min
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
            departure_increments=truth["departure_rate_increments_per_second"][
                start:stop
            ],
            departure_linear=truth[
                "departure_linear_references_per_second"
            ][start:stop],
            energy_directions=energy,
            curvature_basis=inputs["curvature_basis"],
        )

    high_start = manifest.NEW_TRAINING_CANDIDATES
    high_stop = high_start + 2 * manifest.NEW_TUNING_DIRECTIONS
    low_start = (
        manifest.NEW_TRAINING_CANDIDATES
        + 2 * manifest.NEW_TUNING_DIRECTIONS
        + manifest.NEW_HOLDOUT_CANDIDATES
    )
    low_stop = manifest.PLANNED_CANDIDATES
    high = targets(high_start, high_stop)
    low = targets(low_start, low_stop)

    def differences(name: str) -> np.ndarray:
        return np.asarray(
            [
                _relative_error(high[name][index], low[name][index])
                for index in range(manifest.NEW_TUNING_DIRECTIONS)
            ],
            dtype=float,
        )

    quadratic = differences("rate_quadratic_targets")
    cubic = differences("rate_cubic_targets")
    curvature = differences("curvature_cubic_targets")
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


def _model_gate_checks(
    model: dict, radial: dict, fit: dict, contract: dict
) -> dict:
    model_gates = contract["binding_independent_model_gates"]
    radial_gates = contract["binding_radial_consistency_gates"]
    design_gates = contract["design_gates"]
    checks = {
        name: model[name] <= threshold
        for name, threshold in model_gates.items()
        if name != "minimum_reconstructed_state_reconstruction_factor"
        and not name.startswith("minimum_reconstructed_scattering")
    }
    checks["minimum_reconstructed_state_reconstruction_factor"] = model[
        "minimum_reconstructed_state_reconstruction_factor"
    ] >= model_gates["minimum_reconstructed_state_reconstruction_factor"]
    checks["minimum_reconstructed_scattering_optical_depth"] = model[
        "minimum_reconstructed_scattering_optical_depth"
    ] >= model_gates["minimum_reconstructed_scattering_optical_depth"]
    checks.update(
        {
            name: radial[name] <= threshold
            for name, threshold in radial_gates.items()
        }
    )
    checks.update(
        {
            "actual_quadratic_rank": fit["actual_quadratic_feature_rank"]
            == design_gates["quadratic_feature_rank_equal"],
            "actual_quadratic_condition": fit[
                "actual_quadratic_feature_condition_number"
            ]
            <= design_gates["quadratic_feature_condition_number_max"],
            "actual_cubic_rank": fit["actual_cubic_feature_rank"]
            == design_gates["cubic_feature_rank_equal"],
            "actual_cubic_condition": fit[
                "actual_cubic_feature_condition_number"
            ]
            <= design_gates["cubic_feature_condition_number_max"],
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
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
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
        raise RuntimeError("tensor rate validation is already canonicalized")
    inputs = _load_inputs()
    truth_metrics, truth_arrays, fit_metrics, fit_arrays = _execute_truth(inputs)
    contract = manifest._contract()
    truth_checks = _truth_gate_checks(
        truth_metrics, contract["binding_truth_rate_gates"]
    )
    model_metrics = {}
    radial_metrics = {}
    model_arrays = {}
    model_checks = {"truth_database_complete": False}
    truth_passed = all(truth_checks.values())
    if truth_passed:
        radial_metrics = _radial_metrics(inputs, truth_arrays)
        model_metrics, model_arrays = _validation_metrics(
            inputs, truth_arrays, fit_arrays
        )
        model_checks = _model_gate_checks(
            model_metrics, radial_metrics, fit_metrics, contract
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
        CANONICAL_DIRECTORY / "tensor_closure.npz",
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
        shutil.copy2(
            FIT_ARRAY_PATH, CANONICAL_DIRECTORY / "frozen_coefficients.npz"
        )
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
            "geometry_hashes": _checksums(parent.CANONICAL_DIRECTORY),
            "manifest_hashes": _checksums(manifest.CANONICAL_DIRECTORY),
            "old_closure_sha256": _sha(OLD_CLOSURE_PATH),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
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
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    rate_text = "not evaluated"
    state_text = "not evaluated"
    if model_metrics:
        rate_text = (
            f"tuning median/max `{model_metrics['tuning_median_full_departure_rate_relative_error']:.6e}` / "
            f"`{model_metrics['tuning_maximum_full_departure_rate_relative_error']:.6e}`; holdout median/max "
            f"`{model_metrics['holdout_median_full_departure_rate_relative_error']:.6e}` / "
            f"`{model_metrics['holdout_maximum_full_departure_rate_relative_error']:.6e}`"
        )
        state_text = (
            f"maximum full-state decoder error `{model_metrics['maximum_full_scaled_state_decoder_relative_error']:.6e}`"
        )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Active-8 tensor rate validation WP10c9d6c7c3b5c4f25bp",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{truth_metrics['completed_nonbase_rate_evaluations']}` of `{manifest.PLANNED_CANDIDATES}` new exact truth-rate evaluations; failures: `{truth_metrics['failed_rate_evaluations']}`.",
                "",
                "The 4,848 nonlinear coefficients were frozen and hashed after training truth and before any new tuning or holdout rate response was read.",
                "",
                f"Full departure-rate validation: {rate_text}.",
                "",
                f"State reconstruction: {state_text}.",
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
