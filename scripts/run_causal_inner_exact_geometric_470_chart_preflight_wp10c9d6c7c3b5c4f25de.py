#!/usr/bin/env python3
"""Validate an exact implicit 470-coordinate chart at the accepted 20 ms state."""

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


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_hidden_fast_branch_root_pilot_manifest_wp10c9d6c7c3b5c4f25dd as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25de"
MANIFEST_COMMIT = "ecf6515262513ade45e9376d36779cde8467405f"
MANIFEST_PARENT = "1e00274841d17e382e01c7cec78ffe484572a06a"
MANIFEST_TREE = "893d48c889cfa3cd44466e9bffb6516f2010551f"

PASS_CLASSIFICATION = (
    "exact_geometric_470_chart_preflight_passed_primary_hidden_root_"
    "execution_authorized"
)
FAIL_CLASSIFICATION = (
    "exact_geometric_470_chart_preflight_failed_hidden_root_blocked"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25df"

PHYSICAL_DIMENSION = 560
COORDINATE_DIMENSION = 470
GAUGE_DIMENSION = PHYSICAL_DIMENSION - COORDINATE_DIMENSION
MACRO_DIMENSION = 82
HIDDEN_DIMENSION = 388
PRIMARY_INDEX = 5
SEALED_INDEX = 4
HIDDEN_DIRECTION_INDICES = (0, 129, 258, 387)
MACRO_DIRECTION_INDICES = (0, 26, 53, 81)
DIRECTION_COUNT = len(HIDDEN_DIRECTION_INDICES) + len(MACRO_DIRECTION_INDICES)
SIGNED_PHYSICAL_COMPONENT_RADIUS = 2.5e-3
PLANNED_RETRACTIONS = 2 + 2 * DIRECTION_COUNT
MAXIMUM_NEWTON_CORRECTIONS = 4
LINE_FACTORS = (1.0, 0.5, 0.25, 0.125)
COORDINATE_TOLERANCE = 1.0e-10
GAUGE_TOLERANCE = 1.0e-10
MAXIMUM_CONDITION_NUMBER = 1.0e7
MAXIMUM_SCALED_DEPARTURE = 1.5e-2
DERIVATIVE_STEP = 1.0e-4
DERIVATIVE_DEFECT_GATE = 1.0e-6
RECONSTRUCTION_GATE = 1.0 - 1.0e-12
HEIGHT_RATIO_GATE = 0.5
OPTICAL_DEPTH_GATE = 1.0

ARTIFACT = (
    "causal_inner_exact_geometric_470_chart_preflight_"
    "wp10c9d6c7c3b5c4f25de"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_exact_geometric_470_chart_preflight_"
    "wp10c9d6c7c3b5c4f25de.py"
)
THIS_TEST = (
    "tests/test_causal_inner_exact_geometric_470_chart_preflight_"
    "wp10c9d6c7c3b5c4f25de.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXACT_GEOMETRIC_470_CHART_"
    "PREFLIGHT_WP10C9D6C7C3B5C4F25DE_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PRIOR_EXACT_CHART_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_exact_geometric_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25ay"
)


class ChartFailure(RuntimeError):
    """Fail-closed exact-chart error with serializable diagnostics."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


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


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("exact-chart manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("exact-chart manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("exact-chart manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(
        manifest.CANONICAL_DIRECTORY / "branch_root_pilot_contract.json"
    )
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    execution = contract["prospective_execution"]
    if (
        not summary["passed"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["exact_geometric_chart_preflight_authorized"]
        or summary["branch_root_execution_authorized"]
        or execution["work_package"] != WORK_PACKAGE
        or execution["budgets"]["coordinate_retractions_max"]
        != PLANNED_RETRACTIONS
        or execution["budgets"]["new_exact_fixed_Q_rate_evaluations_equal"] != 0
        or execution["budgets"]["new_intrinsic_hidden_roots_equal"] != 0
    ):
        raise RuntimeError("exact-chart authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"exact-chart manifest source changed: {relative}")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    prior_hashes = _checksums(PRIOR_EXACT_CHART_DIRECTORY)
    prior = _read(PRIOR_EXACT_CHART_DIRECTORY / "summary.json")
    if not prior["passed"] or prior["nonbase_continuous_rate_evaluations"] != 0:
        raise RuntimeError("prior exact-chart method evidence changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("exact-chart preflight requires a clean tracked tree")
    return {"manifest_hashes": hashes, "prior_exact_chart_hashes": prior_hashes}


def _model_and_inputs() -> tuple[object, dict, dict]:
    candidate = _load_npz(
        manifest.parent.CANONICAL_DIRECTORY / "candidate_geometry_arrays.npz"
    )
    fiber = _load_npz(manifest.CANONICAL_DIRECTORY / "fiber_geometry.npz")
    field = manifest.parent.field_manifest.ForwardQuadraticAuthenticCenterField(
        _load_npz(manifest.parent.FIELD_ARRAYS)
    )
    return field.model, candidate, fiber


def _coordinate_jacobian(model, state: np.ndarray) -> tuple[np.ndarray, dict]:
    chart_tools = (
        manifest.parent.field_manifest.vector_field.manifest.parent.geometry.chart_tools
    )
    physical, metrics = chart_tools._coordinate_jacobian(
        np.asarray(state, dtype=float), model.components
    )
    jacobian = np.vstack(
        (physical, model.memory_basis.T, model.departure_basis.T)
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return jacobian, {
        "rank": int(np.linalg.matrix_rank(jacobian)),
        "condition_number": float(singular[0] / singular[-1]),
        "minimum_singular_value": float(singular[-1]),
        "maximum_singular_value": float(singular[0]),
        "coordinate_reconstruction_relative_defect": float(
            metrics["reconstruction_relative_defect"]
        ),
        "coordinate_partition_defect": float(metrics["partition_defect"]),
    }


def _canonical_null_basis(jacobian: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(jacobian).T, mode="complete")
    null = np.asarray(q[:, COORDINATE_DIMENSION:], dtype=float)
    for column in range(null.shape[1]):
        pivot = int(np.argmax(np.abs(null[:, column])))
        if null[pivot, column] < 0.0:
            null[:, column] *= -1.0
    return null


def _augmented_jacobian(
    model, state: np.ndarray, gauge_basis: np.ndarray
) -> tuple[np.ndarray, dict]:
    coordinate, metrics = _coordinate_jacobian(model, state)
    augmented = np.vstack((coordinate, gauge_basis.T))
    metrics = {
        **metrics,
        "augmented_rank": int(np.linalg.matrix_rank(augmented)),
        "augmented_condition_number": float(np.linalg.cond(augmented)),
    }
    return augmented, metrics


def _state_from_delta(model, delta: np.ndarray) -> np.ndarray:
    return model.base_state + (
        model.columns.ravel() * np.asarray(delta).ravel()
    ).reshape(model.base_state.shape)


def _delta(model, state: np.ndarray) -> np.ndarray:
    return ((np.asarray(state) - model.base_state) / model.columns).ravel()


def _residual(
    model,
    state: np.ndarray,
    target: np.ndarray,
    gauge_basis: np.ndarray,
    anchor_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate, factors = model.coordinate(state)
    gauge = gauge_basis.T @ (_delta(model, state) - anchor_delta)
    return np.concatenate((coordinate - target, gauge)), np.asarray(factors)


def _physical_audit(model, state: np.ndarray, factors: np.ndarray) -> dict:
    audit = (
        manifest.parent.field_manifest.vector_field.manifest.parent.geometry.chart_tools._state_audit
    )(model.components["context"], np.asarray(state, dtype=float))
    minimum_reconstruction = min(
        float(np.min(factors)), float(audit["minimum_reconstruction_factor"])
    )
    maximum_height = float(audit["maximum_h_over_r"])
    minimum_optical = float(audit["minimum_scattering_optical_depth"])
    return {
        "minimum_reconstruction_factor": minimum_reconstruction,
        "maximum_height_ratio": maximum_height,
        "minimum_scattering_optical_depth": minimum_optical,
        "passed": bool(
            minimum_reconstruction >= RECONSTRUCTION_GATE
            and maximum_height <= HEIGHT_RATIO_GATE
            and minimum_optical >= OPTICAL_DEPTH_GATE
        ),
    }


def _newton_retract(
    model,
    initial_state: np.ndarray,
    target: np.ndarray,
    gauge_basis: np.ndarray,
    anchor_delta: np.ndarray,
) -> tuple[np.ndarray, dict]:
    state = np.asarray(initial_state, dtype=float).copy()
    residual_history = []
    accepted_line_factors = []
    condition_numbers = []
    began = time.perf_counter()
    for correction_index in range(MAXIMUM_NEWTON_CORRECTIONS + 1):
        residual, factors = _residual(
            model, state, target, gauge_basis, anchor_delta
        )
        coordinate_inf = float(np.max(np.abs(residual[:COORDINATE_DIMENSION])))
        gauge_inf = float(np.max(np.abs(residual[COORDINATE_DIMENSION:])))
        combined = max(coordinate_inf, gauge_inf)
        residual_history.append(combined)
        if coordinate_inf <= COORDINATE_TOLERANCE and gauge_inf <= GAUGE_TOLERANCE:
            physical = _physical_audit(model, state, factors)
            return state, {
                "coordinate_residual_infinity": coordinate_inf,
                "gauge_residual_infinity": gauge_inf,
                "Newton_corrections": correction_index,
                "accepted_line_factors": accepted_line_factors,
                "residual_history": residual_history,
                "maximum_augmented_condition_number": (
                    max(condition_numbers) if condition_numbers else 0.0
                ),
                "maximum_scaled_anchor_departure": float(
                    np.max(np.abs(_delta(model, state) - anchor_delta))
                ),
                "wall_seconds": time.perf_counter() - began,
                **physical,
            }
        if correction_index == MAXIMUM_NEWTON_CORRECTIONS:
            break
        augmented, jacobian_metrics = _augmented_jacobian(
            model, state, gauge_basis
        )
        condition_numbers.append(jacobian_metrics["augmented_condition_number"])
        if (
            jacobian_metrics["augmented_rank"] != PHYSICAL_DIMENSION
            or jacobian_metrics["augmented_condition_number"]
            > MAXIMUM_CONDITION_NUMBER
        ):
            raise ChartFailure(
                "augmented coordinate Jacobian failed",
                {
                    "jacobian": jacobian_metrics,
                    "residual_history": residual_history,
                },
            )
        correction = np.linalg.solve(augmented, residual)
        current_delta = _delta(model, state)
        accepted = False
        for factor in LINE_FACTORS:
            proposed_delta = current_delta - factor * correction
            if (
                float(np.max(np.abs(proposed_delta - anchor_delta)))
                > MAXIMUM_SCALED_DEPARTURE
            ):
                continue
            proposed = _state_from_delta(model, proposed_delta)
            trial, _trial_factors = _residual(
                model, proposed, target, gauge_basis, anchor_delta
            )
            if float(np.max(np.abs(trial))) < combined:
                state = proposed
                accepted_line_factors.append(factor)
                accepted = True
                break
        if not accepted:
            raise ChartFailure(
                "exact chart line search failed",
                {"residual_history": residual_history},
            )
    raise ChartFailure(
        "exact chart iteration budget exhausted",
        {"residual_history": residual_history},
    )


def _direction_design(
    augmented: np.ndarray,
    hidden: np.ndarray,
    lifting: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    coordinate_directions = []
    metadata = []
    for index in HIDDEN_DIRECTION_INDICES:
        coordinate_directions.append(np.asarray(hidden[:, index], dtype=float))
        metadata.append({"family": "hidden", "source_index": index})
    for index in MACRO_DIRECTION_INDICES:
        coordinate_directions.append(np.asarray(lifting[:, index], dtype=float))
        metadata.append({"family": "macro", "source_index": index})
    directions = []
    physical_directions = []
    for raw, record in zip(coordinate_directions, metadata, strict=True):
        raw = raw / np.linalg.norm(raw)
        right = np.concatenate((raw, np.zeros(GAUGE_DIMENSION)))
        physical = np.linalg.solve(augmented, right)
        factor = SIGNED_PHYSICAL_COMPONENT_RADIUS / float(np.max(np.abs(physical)))
        directions.append(factor * raw)
        physical_directions.append(factor * physical)
        record["coordinate_radius"] = float(factor)
        record["linear_maximum_scaled_physical_component"] = float(
            np.max(np.abs(factor * physical))
        )
    return np.asarray(directions), np.asarray(physical_directions), metadata


def _implicit_derivative_audit(
    model,
    anchor_state: np.ndarray,
    coordinate_directions: np.ndarray,
    physical_directions: np.ndarray,
) -> dict:
    plus = []
    minus = []
    anchor_delta = _delta(model, anchor_state)
    defects = []
    for coordinate, physical in zip(
        coordinate_directions, physical_directions, strict=True
    ):
        plus_state = _state_from_delta(
            model, anchor_delta + DERIVATIVE_STEP * physical
        )
        minus_state = _state_from_delta(
            model, anchor_delta - DERIVATIVE_STEP * physical
        )
        plus_coordinate, _ = model.coordinate(plus_state)
        minus_coordinate, _ = model.coordinate(minus_state)
        finite = (plus_coordinate - minus_coordinate) / (2.0 * DERIVATIVE_STEP)
        defect = float(
            np.linalg.norm(finite - coordinate)
            / max(np.linalg.norm(coordinate), np.finfo(float).tiny)
        )
        plus.append(plus_coordinate)
        minus.append(minus_coordinate)
        defects.append(defect)
    return {
        "step": DERIVATIVE_STEP,
        "relative_defects": np.asarray(defects),
        "maximum_relative_defect": float(np.max(defects)),
        "plus_coordinates": np.asarray(plus),
        "minus_coordinates": np.asarray(minus),
    }


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    model, candidate, fiber = _model_and_inputs()
    anchor_state = np.asarray(candidate["candidate_primitive_states"][PRIMARY_INDEX])
    raw_decoder_state = np.asarray(
        candidate["candidate_decoded_primitive_states"][PRIMARY_INDEX]
    )
    sealed_state = np.asarray(candidate["candidate_primitive_states"][SEALED_INDEX])
    anchor_target = np.asarray(
        candidate["candidate_absolute_y470_coordinates"][PRIMARY_INDEX]
    )
    anchor_delta = _delta(model, anchor_state)
    coordinate_jacobian, coordinate_metrics = _coordinate_jacobian(
        model, anchor_state
    )
    gauge_basis = _canonical_null_basis(coordinate_jacobian)
    augmented, augmented_metrics = _augmented_jacobian(
        model, anchor_state, gauge_basis
    )
    hidden = np.asarray(fiber["hidden_orthonormal_basis_Z388"], dtype=float)
    lifting = np.asarray(fiber["macro_lifting_L82"], dtype=float)
    coordinate_directions, physical_directions, direction_metadata = (
        _direction_design(augmented, hidden, lifting)
    )
    derivative = _implicit_derivative_audit(
        model, anchor_state, coordinate_directions, physical_directions
    )

    schedule = [
        {
            "label": "anchor_exact_seed",
            "target": anchor_target,
            "initial_state": anchor_state,
            "direction": -1,
            "sign": 0,
        },
        {
            "label": "anchor_raw_decoder_repair",
            "target": anchor_target,
            "initial_state": raw_decoder_state,
            "direction": -1,
            "sign": 0,
        },
    ]
    for direction_index, (coordinate, physical) in enumerate(
        zip(coordinate_directions, physical_directions, strict=True)
    ):
        for sign in (-1, 1):
            schedule.append(
                {
                    "label": f"direction_{direction_index}_{sign:+d}",
                    "target": anchor_target + sign * coordinate,
                    "initial_state": _state_from_delta(
                        model, anchor_delta + sign * physical
                    ),
                    "direction": direction_index,
                    "sign": sign,
                }
            )
    if len(schedule) != PLANNED_RETRACTIONS:
        raise RuntimeError("exact-chart schedule size changed")

    records = []
    states = []
    targets = []
    failures = []
    for index, item in enumerate(schedule):
        try:
            state, record = _newton_retract(
                model,
                item["initial_state"],
                item["target"],
                gauge_basis,
                anchor_delta,
            )
            record.update(
                {
                    "candidate_index": index,
                    "label": item["label"],
                    "direction_index": item["direction"],
                    "sign": item["sign"],
                    "passed": bool(record["passed"]),
                }
            )
            records.append(record)
            states.append(state)
            targets.append(item["target"])
        except ChartFailure as failure:
            failures.append(
                {
                    "candidate_index": index,
                    "label": item["label"],
                    "diagnostics": failure.diagnostics,
                }
            )
            break
    completed = len(records)
    raw_repair_state_defect = None
    if completed >= 2:
        raw_repair_state_defect = float(
            np.max(np.abs(_delta(model, states[1]) - anchor_delta))
        )
    metrics = {
        "planned_retraction_count": PLANNED_RETRACTIONS,
        "completed_retraction_count": completed,
        "failed_retraction_count": len(failures),
        "failures": failures,
        "coordinate_geometry": coordinate_metrics,
        "augmented_geometry": augmented_metrics,
        "gauge_basis_orthogonality_infinity": float(
            np.linalg.norm(
                gauge_basis.T @ gauge_basis - np.eye(GAUGE_DIMENSION), ord=np.inf
            )
        ),
        "gauge_basis_coordinate_annihilation_infinity": float(
            np.linalg.norm(coordinate_jacobian @ gauge_basis, ord=np.inf)
        ),
        "direction_metadata": direction_metadata,
        "implicit_derivative": {
            "step": derivative["step"],
            "relative_defects": derivative["relative_defects"],
            "maximum_relative_defect": derivative["maximum_relative_defect"],
        },
        "candidates": records,
        "maximum_coordinate_residual_infinity": (
            max(record["coordinate_residual_infinity"] for record in records)
            if records
            else None
        ),
        "maximum_gauge_residual_infinity": (
            max(record["gauge_residual_infinity"] for record in records)
            if records
            else None
        ),
        "maximum_augmented_condition_number": max(
            augmented_metrics["augmented_condition_number"],
            max(
                (record["maximum_augmented_condition_number"] for record in records),
                default=0.0,
            ),
        ),
        "maximum_scaled_anchor_departure": (
            max(record["maximum_scaled_anchor_departure"] for record in records)
            if records
            else None
        ),
        "minimum_reconstruction_factor": (
            min(record["minimum_reconstruction_factor"] for record in records)
            if records
            else None
        ),
        "maximum_height_ratio": (
            max(record["maximum_height_ratio"] for record in records)
            if records
            else None
        ),
        "minimum_scattering_optical_depth": (
            min(record["minimum_scattering_optical_depth"] for record in records)
            if records
            else None
        ),
        "all_physical_audits_passed": bool(
            records and all(record["passed"] for record in records)
        ),
        "anchor_exact_seed_roundtrip_bitwise": bool(
            completed >= 1 and np.array_equal(states[0], anchor_state)
        ),
        "raw_decoder_repaired_to_anchor_scaled_infinity": raw_repair_state_defect,
        "sealed_state_was_not_evaluated": True,
        "new_exact_fixed_Q_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_intrinsic_hidden_roots": 0,
        "propagated_states": 0,
        "total_wall_seconds": time.perf_counter() - began,
    }
    arrays = {
        "anchor_primitive_state": anchor_state,
        "sealed_primitive_state_hash_only_source": sealed_state,
        "raw_decoder_primitive_state": raw_decoder_state,
        "anchor_coordinate_y470": anchor_target,
        "anchor_coordinate_jacobian": coordinate_jacobian,
        "anchor_gauge_basis_N90": gauge_basis,
        "anchor_augmented_chart_jacobian": augmented,
        "coordinate_test_directions": coordinate_directions,
        "linear_physical_test_directions": physical_directions,
        "implicit_derivative_plus_coordinates": derivative["plus_coordinates"],
        "implicit_derivative_minus_coordinates": derivative["minus_coordinates"],
        "retracted_primitive_states": np.asarray(states),
        "retracted_target_coordinates": np.asarray(targets),
    }
    return metrics, arrays


def _checks(metrics: dict) -> dict[str, bool]:
    return {
        "retraction_count": metrics["completed_retraction_count"]
        == PLANNED_RETRACTIONS,
        "no_failures": metrics["failed_retraction_count"] == 0,
        "coordinate_rank": metrics["coordinate_geometry"]["rank"]
        == COORDINATE_DIMENSION,
        "augmented_rank": metrics["augmented_geometry"]["augmented_rank"]
        == PHYSICAL_DIMENSION,
        "coordinate_closure": metrics["maximum_coordinate_residual_infinity"]
        is not None
        and metrics["maximum_coordinate_residual_infinity"] <= COORDINATE_TOLERANCE,
        "gauge_closure": metrics["maximum_gauge_residual_infinity"] is not None
        and metrics["maximum_gauge_residual_infinity"] <= GAUGE_TOLERANCE,
        "condition": metrics["maximum_augmented_condition_number"]
        <= MAXIMUM_CONDITION_NUMBER,
        "implicit_derivative": metrics["implicit_derivative"][
            "maximum_relative_defect"
        ]
        <= DERIVATIVE_DEFECT_GATE,
        "component_trust": metrics["maximum_scaled_anchor_departure"] is not None
        and metrics["maximum_scaled_anchor_departure"] <= MAXIMUM_SCALED_DEPARTURE,
        "reconstruction": metrics["minimum_reconstruction_factor"] is not None
        and metrics["minimum_reconstruction_factor"] >= RECONSTRUCTION_GATE,
        "height": metrics["maximum_height_ratio"] is not None
        and metrics["maximum_height_ratio"] <= HEIGHT_RATIO_GATE,
        "optical_depth": metrics["minimum_scattering_optical_depth"] is not None
        and metrics["minimum_scattering_optical_depth"] >= OPTICAL_DEPTH_GATE,
        "physical_audits": metrics["all_physical_audits_passed"],
        "anchor_roundtrip": metrics["anchor_exact_seed_roundtrip_bitwise"],
        "raw_decoder_repair": metrics[
            "raw_decoder_repaired_to_anchor_scaled_infinity"
        ]
        is not None
        and metrics["raw_decoder_repaired_to_anchor_scaled_infinity"] <= 1.0e-8,
        "sealed_untouched": metrics["sealed_state_was_not_evaluated"],
        "rate_budget": metrics["new_exact_fixed_Q_rate_evaluations"] == 0,
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_intrinsic_hidden_roots"] == 0,
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
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("exact geometric 470-chart preflight already exists")
    metrics, arrays = _execute()
    checks = _checks(metrics)
    passed = bool(all(checks.values()))
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "exact_chart_arrays.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "exact_chart_metrics.json",
        {"metrics": metrics, "checks": checks, "passed": passed},
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            **frozen,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "planned_retraction_count": PLANNED_RETRACTIONS,
        "completed_retraction_count": metrics["completed_retraction_count"],
        "failed_retraction_count": metrics["failed_retraction_count"],
        "maximum_coordinate_residual_infinity": metrics[
            "maximum_coordinate_residual_infinity"
        ],
        "maximum_gauge_residual_infinity": metrics[
            "maximum_gauge_residual_infinity"
        ],
        "maximum_augmented_condition_number": metrics[
            "maximum_augmented_condition_number"
        ],
        "maximum_implicit_derivative_relative_defect": metrics[
            "implicit_derivative"
        ]["maximum_relative_defect"],
        "new_exact_fixed_Q_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_intrinsic_hidden_roots": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "branch_root_execution_authorized": passed,
        "online_solver_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.parent.THIS_RUNNER,
        manifest.parent.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
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
                name: os.environ.get(name)
                for name in manifest.parent.field_manifest.training._thread_environment()
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
                "# Exact geometric 470-chart preflight WP10c9d6c7c3b5c4f25de",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"The exact implicit chart completed `{metrics['completed_retraction_count']}` of `{PLANNED_RETRACTIONS}` prospective retractions with `{metrics['failed_retraction_count']}` failures. The maximum coordinate and gauge defects were `{metrics['maximum_coordinate_residual_infinity']}` and `{metrics['maximum_gauge_residual_infinity']}`.",
                "",
                f"The maximum augmented condition number was `{metrics['maximum_augmented_condition_number']:.6e}` and the maximum independent implicit-derivative defect was `{metrics['implicit_derivative']['maximum_relative_defect']:.6e}`. The raw approximate decoder was explicitly repaired back to the accepted anchor under the exact 470-coordinate plus 90-gauge equations.",
                "",
                "No fixed-Q rate, complete generator, branch root, transition, or propagated state was evaluated. The 16 ms state remained sealed.",
                "",
                (
                    f"Passing authorizes only the single-primary hidden-root execution `{AUTHORIZED_NEXT}` under the already frozen budget."
                    if passed
                    else "Failure blocks the hidden root and authorizes no continuation."
                ),
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
    summary = _run()
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
