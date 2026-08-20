#!/usr/bin/env python3
"""Diagnose a full-departure dual quadratic/cubic rate architecture."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_active8_projective_kernel_rate_validation_wp10c9d6c7c3b5c4f25bt as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bu"
PARENT_COMMIT = "059ebd862a037176b327408f973278ef6cb24c1a"
PARENT_PARENT = "e0a161bbf3ea1e1dbe9eb30c2c89db8fbbe41a85"
PARENT_TREE = "7ff734b49aa00916874b1d940b1d1f47d15cb9a2"

PASS_CLASSIFICATION = (
    "departure28_dual_polynomial_architecture_selected_for_"
    "independent_validation"
)
FAIL_CLASSIFICATION = "departure28_dual_polynomial_architecture_not_selected"
AUTHORIZED_NEXT = (
    "definitions_only_departure28_dual_polynomial_independent_"
    "validation_manifest"
)

ARTIFACT = (
    "causal_inner_departure28_dual_polynomial_diagnosis_"
    "wp10c9d6c7c3b5c4f25bu"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_departure28_dual_polynomial_diagnosis_"
    "wp10c9d6c7c3b5c4f25bu.py"
)
THIS_TEST = (
    "tests/test_causal_inner_departure28_dual_polynomial_diagnosis_"
    "wp10c9d6c7c3b5c4f25bu.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_DEPARTURE28_DUAL_POLYNOMIAL_"
    "DIAGNOSIS_WP10C9D6C7C3B5C4F25BU_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

OLD_CLOSURE_PATH = parent.previous.OLD_CLOSURE_PATH
TENSOR_CLOSURE_PATH = parent.PREVIOUS_CLOSURE_PATH
PROJECTIVE_CLOSURE_PATH = parent.CANONICAL_DIRECTORY / "projective_kernel_closure.npz"
PROJECTIVE_COEFFICIENT_PATH = parent.CANONICAL_DIRECTORY / "frozen_coefficients.npz"

DEPARTURE_DIMENSION = 28
REVEALED_DIRECTION_COUNT = 160
RETROSPECTIVE_TRAINING_DIRECTION_COUNT = 144
RETROSPECTIVE_VALIDATION_DIRECTION_COUNT = 16
EVEN_KERNEL_POWER = 2
ODD_KERNEL_POWER = 3
EVEN_TARGET_WEIGHT_EXPONENT = 1.0
ODD_TARGET_WEIGHT_EXPONENT = 0.0
EVEN_TIKHONOV_REGULARIZATION = 10.0 ** -5.5
ODD_TIKHONOV_REGULARIZATION = 1.0e-6
RATE_COEFFICIENT_COUNT = 2 * REVEALED_DIRECTION_COUNT * DEPARTURE_DIMENSION
CURVATURE_COEFFICIENT_COUNT = 120 * 4
TOTAL_NONLINEAR_COEFFICIENT_COUNT = (
    RATE_COEFFICIENT_COUNT + CURVATURE_COEFFICIENT_COUNT
)


_plain = parent._plain
_read = parent._read
_write_json = parent._write_json
_write_npz = parent._write_npz
_load_npz = parent._load_npz
_sha = parent._sha
_checksums = parent._checksums


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("projective-kernel rejection commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("projective-kernel rejection lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("projective-kernel rejection tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    failed_model_checks = sorted(
        name for name, passed in metrics["model_checks"].items() if not passed
    )
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or not summary["truth_database_passed"]
        or summary["completed_nonbase_rate_evaluations"] != 48
        or summary["failed_rate_evaluations"] != 0
        or not all(metrics["truth_checks"].values())
        or failed_model_checks
        != [
            "holdout_maximum_full_departure_rate_relative_error",
            "holdout_maximum_nonlinear_departure_rate_relative_error",
        ]
    ):
        raise RuntimeError("projective-kernel rejection changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"projective-kernel source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("departure-28 diagnosis requires a clean tracked tree")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _pair_targets(
    coordinates: np.ndarray,
    departure_rates: np.ndarray,
    linear_rates: np.ndarray,
) -> dict[str, np.ndarray]:
    if coordinates.shape[0] % 2:
        raise RuntimeError("signed departure-28 data are not paired")
    nonlinear = departure_rates - linear_rates
    directions = []
    radii = []
    quadratic = []
    cubic = []
    for negative in range(0, coordinates.shape[0], 2):
        positive = negative + 1
        odd_coordinate = 0.5 * (
            coordinates[positive] - coordinates[negative]
        )
        radius = float(np.linalg.norm(odd_coordinate))
        if radius <= np.finfo(float).tiny:
            raise RuntimeError("departure-28 pair radius vanished")
        directions.append(odd_coordinate / radius)
        radii.append(radius)
        quadratic.append(
            0.5 * (nonlinear[positive] + nonlinear[negative]) / radius**2
        )
        cubic.append(
            0.5 * (nonlinear[positive] - nonlinear[negative]) / radius**3
        )
    return {
        "directions": np.asarray(directions, dtype=float),
        "radii": np.asarray(radii, dtype=float),
        "quadratic_targets": np.asarray(quadratic, dtype=float),
        "cubic_targets": np.asarray(cubic, dtype=float),
    }


def _revealed_database() -> dict[str, np.ndarray]:
    old = _load_npz(OLD_CLOSURE_PATH)
    tensor = _load_npz(TENSOR_CLOSURE_PATH)
    projective = _load_npz(PROJECTIVE_CLOSURE_PATH)
    chunks = (
        (0, old, slice(0, 112)),
        (1, tensor, slice(0, 176)),
        (2, projective, slice(0, 32)),
    )
    coordinates = np.vstack(
        [data["candidate_departure_coordinates"][selection] for _, data, selection in chunks]
    )
    departure_rates = np.vstack(
        [data["departure_rate_increments_per_second"][selection] for _, data, selection in chunks]
    )
    linear_rates = np.vstack(
        [data["departure_linear_references_per_second"][selection] for _, data, selection in chunks]
    )
    source_codes = np.concatenate(
        [
            np.full(data["candidate_departure_coordinates"][selection].shape[0], code, dtype=int)
            for code, data, selection in chunks
        ]
    )
    targets = _pair_targets(coordinates, departure_rates, linear_rates)
    if (
        coordinates.shape != (2 * REVEALED_DIRECTION_COUNT, DEPARTURE_DIMENSION)
        or departure_rates.shape != coordinates.shape
        or linear_rates.shape != coordinates.shape
        or targets["directions"].shape
        != (REVEALED_DIRECTION_COUNT, DEPARTURE_DIMENSION)
        or source_codes.shape != (2 * REVEALED_DIRECTION_COUNT,)
    ):
        raise RuntimeError("revealed departure-28 database dimensions changed")
    return {
        "signed_departure_coordinates": coordinates,
        "signed_departure_rate_increments_per_second": departure_rates,
        "signed_linear_rate_references_per_second": linear_rates,
        "signed_source_codes": source_codes,
        **targets,
    }


def _kernel(left: np.ndarray, right: np.ndarray, power: int) -> np.ndarray:
    return (np.asarray(left, dtype=float) @ np.asarray(right, dtype=float).T) ** power


def _fit(
    directions: np.ndarray,
    targets: np.ndarray,
    *,
    power: int,
    weight_exponent: float,
    regularization: float,
) -> tuple[np.ndarray, dict]:
    norms = np.linalg.norm(targets, axis=1)
    scale = float(np.median(norms))
    weights = (
        scale / np.maximum(norms, np.finfo(float).tiny)
    ) ** weight_exponent
    system = _kernel(directions, directions, power) + regularization * np.diag(
        1.0 / weights
    )
    coefficients = np.linalg.solve(system, targets)
    metrics = {
        "rank": int(np.linalg.matrix_rank(system)),
        "condition_number": float(np.linalg.cond(system)),
        "target_norm_median": scale,
        "target_weight_minimum": float(np.min(weights)),
        "target_weight_maximum": float(np.max(weights)),
    }
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("dual polynomial coefficients are nonfinite")
    return coefficients, metrics


def _predict(
    coordinates: np.ndarray,
    centers: np.ndarray,
    even_coefficients: np.ndarray,
    odd_coefficients: np.ndarray,
) -> np.ndarray:
    radii = np.linalg.norm(coordinates, axis=1)
    if np.any(radii <= np.finfo(float).tiny):
        raise RuntimeError("departure-28 prediction radius vanished")
    unit = coordinates / radii[:, None]
    return (
        radii[:, None] ** 2
        * (_kernel(unit, centers, EVEN_KERNEL_POWER) @ even_coefficients)
        + radii[:, None] ** 3
        * (_kernel(unit, centers, ODD_KERNEL_POWER) @ odd_coefficients)
    )


def _errors(
    prediction: np.ndarray,
    truth: np.ndarray,
    linear: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nonlinear = truth - linear
    nonlinear_error = np.linalg.norm(prediction - nonlinear, axis=1) / np.maximum(
        np.linalg.norm(nonlinear, axis=1), np.finfo(float).tiny
    )
    full_error = np.linalg.norm(linear + prediction - truth, axis=1) / np.maximum(
        np.linalg.norm(truth, axis=1), np.finfo(float).tiny
    )
    return nonlinear_error, full_error


def _aggregates(nonlinear: np.ndarray, full: np.ndarray) -> dict:
    return {
        "median_nonlinear_departure_rate_relative_error": float(np.median(nonlinear)),
        "p95_nonlinear_departure_rate_relative_error": float(np.quantile(nonlinear, 0.95)),
        "maximum_nonlinear_departure_rate_relative_error": float(np.max(nonlinear)),
        "median_full_departure_rate_relative_error": float(np.median(full)),
        "p95_full_departure_rate_relative_error": float(np.quantile(full, 0.95)),
        "maximum_full_departure_rate_relative_error": float(np.max(full)),
        "maximum_nonlinear_candidate_index": int(np.argmax(nonlinear)),
        "maximum_full_candidate_index": int(np.argmax(full)),
    }


def _retrospective_validation(database: dict) -> tuple[dict, dict[str, np.ndarray]]:
    count = RETROSPECTIVE_TRAINING_DIRECTION_COUNT
    centers = database["directions"][:count]
    even, even_fit = _fit(
        centers,
        database["quadratic_targets"][:count],
        power=EVEN_KERNEL_POWER,
        weight_exponent=EVEN_TARGET_WEIGHT_EXPONENT,
        regularization=EVEN_TIKHONOV_REGULARIZATION,
    )
    odd, odd_fit = _fit(
        centers,
        database["cubic_targets"][:count],
        power=ODD_KERNEL_POWER,
        weight_exponent=ODD_TARGET_WEIGHT_EXPONENT,
        regularization=ODD_TIKHONOV_REGULARIZATION,
    )
    start = 2 * count
    coordinates = database["signed_departure_coordinates"][start:]
    truth = database["signed_departure_rate_increments_per_second"][start:]
    linear = database["signed_linear_rate_references_per_second"][start:]
    prediction = _predict(coordinates, centers, even, odd)
    nonlinear_error, full_error = _errors(prediction, truth, linear)
    metrics = {
        "diagnostic_only_revealed_validation": True,
        "training_direction_count": count,
        "revealed_validation_direction_count": RETROSPECTIVE_VALIDATION_DIRECTION_COUNT,
        "revealed_validation_candidate_count": int(coordinates.shape[0]),
        "even_fit": even_fit,
        "odd_fit": odd_fit,
        **_aggregates(nonlinear_error, full_error),
    }
    return metrics, {
        "retrospective_even_coefficients": even,
        "retrospective_odd_coefficients": odd,
        "retrospective_predicted_nonlinear_rates_per_second": prediction,
        "retrospective_nonlinear_relative_errors": nonlinear_error,
        "retrospective_full_relative_errors": full_error,
    }


def _loo_predictions(
    directions: np.ndarray,
    signed_coordinates: np.ndarray,
    targets: np.ndarray,
    *,
    power: int,
    weight_exponent: float,
    regularization: float,
) -> tuple[np.ndarray, dict]:
    norms = np.linalg.norm(targets, axis=1)
    scale = float(np.median(norms))
    weights = (
        scale / np.maximum(norms, np.finfo(float).tiny)
    ) ** weight_exponent
    system = _kernel(directions, directions, power) + regularization * np.diag(
        1.0 / weights
    )
    inverse = np.linalg.inv(system)
    coefficients = inverse @ targets
    radii = np.linalg.norm(signed_coordinates, axis=1)
    unit = signed_coordinates / radii[:, None]
    candidate_kernel = _kernel(unit, directions, power)
    prediction = candidate_kernel @ coefficients
    influence = candidate_kernel @ inverse
    rows = np.arange(signed_coordinates.shape[0])
    pair_indices = rows // 2
    prediction -= (
        influence[rows, pair_indices, None]
        / np.diag(inverse)[pair_indices, None]
        * coefficients[pair_indices]
    )
    return prediction, {
        "rank": int(np.linalg.matrix_rank(system)),
        "condition_number": float(np.linalg.cond(system)),
        "target_norm_median": scale,
        "target_weight_minimum": float(np.min(weights)),
        "target_weight_maximum": float(np.max(weights)),
    }


def _leave_one_direction_out(database: dict) -> tuple[dict, dict[str, np.ndarray]]:
    coordinates = database["signed_departure_coordinates"]
    radii = np.linalg.norm(coordinates, axis=1)
    even_unit, even_fit = _loo_predictions(
        database["directions"],
        coordinates,
        database["quadratic_targets"],
        power=EVEN_KERNEL_POWER,
        weight_exponent=EVEN_TARGET_WEIGHT_EXPONENT,
        regularization=EVEN_TIKHONOV_REGULARIZATION,
    )
    odd_unit, odd_fit = _loo_predictions(
        database["directions"],
        coordinates,
        database["cubic_targets"],
        power=ODD_KERNEL_POWER,
        weight_exponent=ODD_TARGET_WEIGHT_EXPONENT,
        regularization=ODD_TIKHONOV_REGULARIZATION,
    )
    prediction = radii[:, None] ** 2 * even_unit + radii[:, None] ** 3 * odd_unit
    nonlinear_error, full_error = _errors(
        prediction,
        database["signed_departure_rate_increments_per_second"],
        database["signed_linear_rate_references_per_second"],
    )
    metrics = {
        "cross_validation_kind": "leave_one_signed_direction_pair_out",
        "direction_count": REVEALED_DIRECTION_COUNT,
        "candidate_count": 2 * REVEALED_DIRECTION_COUNT,
        "even_fit": even_fit,
        "odd_fit": odd_fit,
        **_aggregates(nonlinear_error, full_error),
    }
    return metrics, {
        "loo_predicted_nonlinear_rates_per_second": prediction,
        "loo_nonlinear_relative_errors": nonlinear_error,
        "loo_full_relative_errors": full_error,
    }


def _diagnose() -> tuple[dict, dict[str, np.ndarray]]:
    database = _revealed_database()
    retrospective, retrospective_arrays = _retrospective_validation(database)
    loo, loo_arrays = _leave_one_direction_out(database)
    even, even_fit = _fit(
        database["directions"],
        database["quadratic_targets"],
        power=EVEN_KERNEL_POWER,
        weight_exponent=EVEN_TARGET_WEIGHT_EXPONENT,
        regularization=EVEN_TIKHONOV_REGULARIZATION,
    )
    odd, odd_fit = _fit(
        database["directions"],
        database["cubic_targets"],
        power=ODD_KERNEL_POWER,
        weight_exponent=ODD_TARGET_WEIGHT_EXPONENT,
        regularization=ODD_TIKHONOV_REGULARIZATION,
    )
    decoder = _load_npz(PROJECTIVE_COEFFICIENT_PATH)[
        "curvature_cubic_coefficients"
    ]
    gates = {
        "median_nonlinear_departure_rate_relative_error": 0.10,
        "p95_nonlinear_departure_rate_relative_error": 0.10,
        "maximum_nonlinear_departure_rate_relative_error": 0.25,
        "median_full_departure_rate_relative_error": 0.02,
        "p95_full_departure_rate_relative_error": 0.02,
        "maximum_full_departure_rate_relative_error": 0.05,
        "even_system_condition_number": 1.0e8,
        "odd_system_condition_number": 1.0e7,
    }
    checks = {}
    for prefix, source in (("retrospective", retrospective), ("loo", loo)):
        for name in (
            "median_nonlinear_departure_rate_relative_error",
            "p95_nonlinear_departure_rate_relative_error",
            "maximum_nonlinear_departure_rate_relative_error",
            "median_full_departure_rate_relative_error",
            "p95_full_departure_rate_relative_error",
            "maximum_full_departure_rate_relative_error",
        ):
            checks[f"{prefix}_{name}"] = source[name] <= gates[name]
    checks.update(
        {
            "retrospective_even_condition": retrospective["even_fit"]["condition_number"]
            <= gates["even_system_condition_number"],
            "retrospective_odd_condition": retrospective["odd_fit"]["condition_number"]
            <= gates["odd_system_condition_number"],
            "loo_even_condition": loo["even_fit"]["condition_number"]
            <= gates["even_system_condition_number"],
            "loo_odd_condition": loo["odd_fit"]["condition_number"]
            <= gates["odd_system_condition_number"],
            "refit_even_condition": even_fit["condition_number"]
            <= gates["even_system_condition_number"],
            "refit_odd_condition": odd_fit["condition_number"]
            <= gates["odd_system_condition_number"],
            "refit_even_rank": even_fit["rank"] == REVEALED_DIRECTION_COUNT,
            "refit_odd_rank": odd_fit["rank"] == REVEALED_DIRECTION_COUNT,
            "frozen_rank4_decoder_shape": decoder.shape == (120, 4),
            "no_dynamic_state_augmentation": True,
            "zero_new_truth_evaluations": True,
        }
    )
    metrics = {
        "diagnostic_only": True,
        "architecture_selected_after_revealed_parent_validation": True,
        "departure_input_dimension": DEPARTURE_DIMENSION,
        "dynamic_state_dimension": 470,
        "stable_descriptor_kernel_dimension": 280,
        "dynamic_curvature_augmentation": False,
        "even_kernel": "dot_squared",
        "odd_kernel": "dot_cubed",
        "even_target_weight_exponent": EVEN_TARGET_WEIGHT_EXPONENT,
        "odd_target_weight_exponent": ODD_TARGET_WEIGHT_EXPONENT,
        "even_Tikhonov_regularization": EVEN_TIKHONOV_REGULARIZATION,
        "odd_Tikhonov_regularization": ODD_TIKHONOV_REGULARIZATION,
        "revealed_direction_count": REVEALED_DIRECTION_COUNT,
        "stored_rate_coefficient_count_after_refit": RATE_COEFFICIENT_COUNT,
        "stored_curvature_coefficient_count": CURVATURE_COEFFICIENT_COUNT,
        "stored_total_nonlinear_coefficient_count": TOTAL_NONLINEAR_COEFFICIENT_COUNT,
        "online_truth_calls_per_macrostep": 0,
        "online_Newton_retractions_per_macrostep": 0,
        "retrospective_validation": retrospective,
        "leave_one_direction_out": loo,
        "all_revealed_refit": {"even_fit": even_fit, "odd_fit": odd_fit},
    }
    return {"metrics": metrics, "gates": gates, "checks": checks}, {
        **database,
        **retrospective_arrays,
        **loo_arrays,
        "all_revealed_even_coefficients": even,
        "all_revealed_odd_coefficients": odd,
        "frozen_rank4_curvature_decoder_coefficients": decoder,
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
                    "scientific_status": "DIAGNOSTIC_ONLY",
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
        raise RuntimeError("departure-28 diagnosis already canonicalized")
    diagnosis, arrays = _diagnose()
    passed = all(diagnosis["checks"].values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = AUTHORIZED_NEXT if passed else None
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", diagnosis)
    _write_npz(CANONICAL_DIRECTORY / "departure28_architecture.npz", arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "diagnostic_only": True,
        "new_truth_evaluations": 0,
        "revealed_direction_count": REVEALED_DIRECTION_COUNT,
        "dynamic_state_dimension": 470,
        "departure_input_dimension": DEPARTURE_DIMENSION,
        "stored_total_nonlinear_coefficient_count": TOTAL_NONLINEAR_COEFFICIENT_COUNT,
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
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "old_closure_sha256": _sha(OLD_CLOSURE_PATH),
            "tensor_closure_sha256": _sha(TENSOR_CLOSURE_PATH),
            "projective_closure_sha256": _sha(PROJECTIVE_CLOSURE_PATH),
        },
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DIAGNOSTIC_ONLY",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
                parent.THIS_RUNNER: _sha(ROOT / parent.THIS_RUNNER),
                parent.THIS_TEST: _sha(ROOT / parent.THIS_TEST),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    retrospective = diagnosis["metrics"]["retrospective_validation"]
    loo = diagnosis["metrics"]["leave_one_direction_out"]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Departure-28 dual-polynomial diagnosis WP10c9d6c7c3b5c4f25bu",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "The selected closure uses the full existing 28-dimensional departure state with an even dot-squared kernel and an odd dot-cubed kernel. It adds no dynamic variable.",
                "",
                f"Revealed 144-to-16 diagnostic validation nonlinear median/max: `{retrospective['median_nonlinear_departure_rate_relative_error']:.6e}` / `{retrospective['maximum_nonlinear_departure_rate_relative_error']:.6e}`; full median/max: `{retrospective['median_full_departure_rate_relative_error']:.6e}` / `{retrospective['maximum_full_departure_rate_relative_error']:.6e}`.",
                "",
                f"Leave-one-direction-out nonlinear median/p95/max: `{loo['median_nonlinear_departure_rate_relative_error']:.6e}` / `{loo['p95_nonlinear_departure_rate_relative_error']:.6e}` / `{loo['maximum_nonlinear_departure_rate_relative_error']:.6e}`; full median/p95/max: `{loo['median_full_departure_rate_relative_error']:.6e}` / `{loo['p95_full_departure_rate_relative_error']:.6e}` / `{loo['maximum_full_departure_rate_relative_error']:.6e}`.",
                "",
                "This is diagnostic-only because the architecture was selected after the parent holdout was revealed.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No trajectory, predictive cycle, or reduced slow evolution is authorized.",
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
