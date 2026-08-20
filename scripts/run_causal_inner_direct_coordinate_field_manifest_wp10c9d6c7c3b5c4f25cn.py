#!/usr/bin/env python3
"""Freeze a direct online coordinate field without state-Jacobian rebuilds."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

import run_causal_inner_certified_0p015_departure_rate_screen_wp10c9d6c7c3b5c4f25cf as original_rate  # noqa: E402
import run_causal_inner_compensated_decoder_geometry_preflight_wp10c9d6c7c3b5c4f25cm as parent  # noqa: E402
import run_causal_inner_shell_gated_atlas_rate_validation_wp10c9d6c7c3b5c4f25ck as mixed_rate  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cn"
PARENT_COMMIT = "2ecc74e1a7367b4d1e1e2070444ee0e22570cb61"
PARENT_PARENT = "07a745928287125b7e73b47f8ba8b022a35b789e"
PARENT_TREE = "f2a306be649f92c1ad2884deee28fcd956b938af"
CLASSIFICATION = "direct_470_coordinate_field_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25co"

Q_KERNEL_LENGTH_SCALE = 1.0e-2
Q_KERNEL_REGULARIZATION = 1.0e-8
PHYSICAL_DIMENSION = 162
MEMORY_DIMENSION = 280
DEPARTURE_DIMENSION = 28
ONLINE_DIMENSION = 470
PLANNED_RATE_EVALUATIONS = 8

ARTIFACT = (
    "causal_inner_direct_coordinate_field_manifest_"
    "wp10c9d6c7c3b5c4f25cn"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_direct_coordinate_field_manifest_"
    "wp10c9d6c7c3b5c4f25cn.py"
)
THIS_TEST = (
    "tests/test_causal_inner_direct_coordinate_field_manifest_"
    "wp10c9d6c7c3b5c4f25cn.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_direct_coordinate_field_validation_"
    "wp10c9d6c7c3b5c4f25co.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_direct_coordinate_field_validation_"
    "wp10c9d6c7c3b5c4f25co.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_DIRECT_COORDINATE_"
    "FIELD_MANIFEST_WP10C9D6C7C3B5C4F25CN_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

OLD_EXTENSION = parent.manifest.OLD_EXTENSION
DECODER_REPAIR = parent.manifest.CANONICAL_DIRECTORY / "compensated_decoder_repair.npz"
ORIGINAL_RATE_ARRAYS = original_rate.CANONICAL_DIRECTORY / "rate_arrays.npz"
MIXED_RATE_ARRAYS = mixed_rate.CANONICAL_DIRECTORY / "rate_arrays.npz"
HOLDOUT_GEOMETRY = parent.CANONICAL_DIRECTORY / "holdout_geometry.npz"
ONLINE_GEOMETRY = original_rate.manifest.ONLINE_GEOMETRY

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


def _relative_rows(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(actual) - np.asarray(expected), axis=1) / np.maximum(
        np.linalg.norm(np.asarray(expected), axis=1), np.finfo(float).tiny
    )


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("compensated decoder geometry commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("compensated decoder geometry lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("compensated decoder geometry tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "geometry_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["classification"] != parent.FULL_CLASSIFICATION
        or summary["largest_passing_component_bound"] != 0.015
        or summary["completed_candidate_count"] != 8
        or summary["failed_candidate_count"] != 0
        or summary["authorized_next"] != "definitions_only_recentered_transition_forecast_manifest"
        or summary["new_truth_rate_calls"] != 0
        or not summary["independently_validated_rate_field_preserved_algebraically"]
        or summary["trajectory_authorized"]
        or not all(
            check
            for rung in metrics["rungs"]
            for check in rung["checks"].values()
        )
    ):
        raise RuntimeError("direct coordinate-field authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"compensated geometry source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("direct coordinate-field manifest requires a clean tracked tree")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _anchored_kernel(
    evaluation: np.ndarray,
    centers: np.ndarray,
    length_scale: float = Q_KERNEL_LENGTH_SCALE,
) -> np.ndarray:
    points = np.asarray(evaluation, dtype=float)
    locations = np.asarray(centers, dtype=float)
    squared = np.sum(
        (points[:, None, :] - locations[None, :, :]) ** 2, axis=2
    )
    kernel = np.exp(-0.5 * squared / float(length_scale) ** 2)
    anchor = np.exp(
        -0.5 * np.sum(locations**2, axis=1) / float(length_scale) ** 2
    )
    return kernel - anchor[None, :]


def _q_correction(
    departure: np.ndarray,
    centers: np.ndarray,
    coefficients: np.ndarray,
    length_scale: float = Q_KERNEL_LENGTH_SCALE,
) -> np.ndarray:
    return np.asarray(
        _anchored_kernel(
            np.asarray(departure, dtype=float).reshape(1, -1),
            centers,
            length_scale,
        )
        @ coefficients,
        dtype=float,
    ).reshape(-1)


class DirectCoordinateField:
    """Cheap 470D field with no state-dependent coordinate-Jacobian build."""

    def __init__(
        self,
        closure: dict[str, np.ndarray],
        *,
        model=None,
        old_extension: dict[str, np.ndarray] | None = None,
        decoder_repair: dict[str, np.ndarray] | None = None,
    ):
        self.model = model or parent.manifest.parent.vector_field.ReducedVectorField()
        self.old_extension = old_extension or _load_npz(OLD_EXTENSION)
        self.decoder_repair = decoder_repair or _load_npz(DECODER_REPAIR)
        self.centers = np.asarray(closure["q_rate_centers"], dtype=float)
        self.coefficients = np.asarray(
            closure["q_rate_coefficients"], dtype=float
        )
        self.restriction = np.asarray(self.model.restriction, dtype=float)

    def _old_shell(self, coordinate: np.ndarray) -> tuple[np.ndarray, float]:
        y = np.asarray(coordinate, dtype=float)
        departure = y[-DEPARTURE_DIMENSION:]
        old_delta = self.model.decoded_delta(y)
        weight = parent.manifest.parent.atlas._shell_weight(
            float(np.max(np.abs(old_delta)))
        )
        extension = parent.manifest.parent.atlas._extension_value(
            departure,
            self.old_extension["extension_center_directions"],
            self.old_extension["decoder_even4_coefficients"],
            self.old_extension["decoder_odd5_coefficients"],
        )
        return old_delta + weight * extension, weight

    def decoded_delta(self, coordinate: np.ndarray) -> np.ndarray:
        y = np.asarray(coordinate, dtype=float)
        departure = y[-DEPARTURE_DIMENSION:]
        old_extended, weight = self._old_shell(y)
        repair = parent.manifest._repair_value(
            departure,
            self.decoder_repair["decoder_repair_centers"],
            self.decoder_repair["decoder_repair_coefficients"],
        )
        return old_extended + weight * repair

    def decoded_state(self, coordinate: np.ndarray) -> np.ndarray:
        delta = self.decoded_delta(coordinate)
        return self.model.base_state + (
            self.model.columns.ravel() * delta
        ).reshape(self.model.base_state.shape)

    def full_state_rate(self, coordinate: np.ndarray) -> np.ndarray:
        y = np.asarray(coordinate, dtype=float)
        departure = y[-DEPARTURE_DIMENSION:]
        old_extended, weight = self._old_shell(y)
        rate_extension = parent.manifest.parent.atlas._extension_value(
            departure,
            self.old_extension["extension_center_directions"],
            self.old_extension["full_state_rate_even4_coefficients"],
            self.old_extension["full_state_rate_odd5_coefficients"],
        )
        return (
            self.model.base_rate
            + self.model.generator @ old_extended
            + self.model.departure_basis
            @ self.model.nonlinear_departure(departure)
            + weight * rate_extension
        )

    def field(self, coordinate: np.ndarray) -> np.ndarray:
        y = np.asarray(coordinate, dtype=float)
        full_rate = self.full_state_rate(y)
        result = self.restriction @ full_rate
        result[:PHYSICAL_DIMENSION] += _q_correction(
            y[-DEPARTURE_DIMENSION:], self.centers, self.coefficients
        )
        return result


def _fit_q_closure() -> tuple[dict[str, np.ndarray], dict]:
    model = parent.manifest.parent.vector_field.ReducedVectorField()
    old = _load_npz(OLD_EXTENSION)
    original = _load_npz(ORIGINAL_RATE_ARRAYS)
    mixed = _load_npz(MIXED_RATE_ARRAYS)
    geometry = _load_npz(ONLINE_GEOMETRY)
    departures = np.vstack(
        (
            old["training_departure_coordinates"],
            mixed["candidate_departure_coordinates"],
        )
    )
    coordinates = np.vstack(
        (
            old["training_online_coordinates"],
            mixed["online_coordinates"],
        )
    )
    predicted_full_rates = np.vstack(
        (
            old["training_extended_full_state_rates_per_second"],
            mixed["predicted_full_state_rates_per_second"],
        )
    )
    exact_online_rates = np.vstack(
        (
            original["online_470_coordinate_rates_per_second"],
            mixed["exact_online_470_coordinate_rates_per_second"],
        )
    )
    restriction = np.asarray(geometry["online_coordinate_restriction"], dtype=float)
    if (
        departures.shape != (16, DEPARTURE_DIMENSION)
        or coordinates.shape != (16, ONLINE_DIMENSION)
        or predicted_full_rates.shape != (16, 560)
        or exact_online_rates.shape != (16, ONLINE_DIMENSION)
        or restriction.shape != (ONLINE_DIMENSION, 560)
        or not np.array_equal(restriction, model.restriction)
    ):
        raise RuntimeError("direct coordinate-field training database changed")
    base_online_rates = (restriction @ predicted_full_rates.T).T
    targets = (
        exact_online_rates[:, :PHYSICAL_DIMENSION]
        - base_online_rates[:, :PHYSICAL_DIMENSION]
    )
    kernel = _anchored_kernel(departures, departures)
    regularized = kernel + Q_KERNEL_REGULARIZATION * np.eye(departures.shape[0])
    coefficients = np.linalg.solve(regularized, targets)
    predicted_online_rates = np.array(base_online_rates, copy=True)
    predicted_online_rates[:, :PHYSICAL_DIMENSION] += kernel @ coefficients
    slices = {
        "full": slice(None),
        "q162": slice(0, PHYSICAL_DIMENSION),
        "z280": slice(PHYSICAL_DIMENSION, PHYSICAL_DIMENSION + MEMORY_DIMENSION),
        "a28": slice(-DEPARTURE_DIMENSION, None),
    }
    errors = {
        name: _relative_rows(
            predicted_online_rates[:, selection],
            exact_online_rates[:, selection],
        )
        for name, selection in slices.items()
    }
    origin_correction = _q_correction(
        np.zeros(DEPARTURE_DIMENSION), departures, coefficients
    )
    direct = DirectCoordinateField(
        {
            "q_rate_centers": departures,
            "q_rate_coefficients": coefficients,
        },
        model=model,
        old_extension=old,
        decoder_repair=_load_npz(DECODER_REPAIR),
    )
    began = time.perf_counter()
    repeated = []
    for coordinate in coordinates:
        repeated.append(direct.field(coordinate))
    direct_wall = time.perf_counter() - began
    repeated = np.asarray(repeated)
    implementation_defect = _relative_rows(repeated, predicted_online_rates)
    metrics = {
        "training_count": 16,
        "kernel_rank": int(np.linalg.matrix_rank(regularized)),
        "kernel_condition_number": float(np.linalg.cond(regularized)),
        "maximum_training_full_coordinate_rate_relative_error": float(
            np.max(errors["full"])
        ),
        "median_training_full_coordinate_rate_relative_error": float(
            np.median(errors["full"])
        ),
        "maximum_training_q162_rate_relative_error": float(
            np.max(errors["q162"])
        ),
        "median_training_q162_rate_relative_error": float(
            np.median(errors["q162"])
        ),
        "maximum_training_z280_rate_relative_error": float(
            np.max(errors["z280"])
        ),
        "maximum_training_a28_rate_relative_error": float(
            np.max(errors["a28"])
        ),
        "origin_q_rate_correction_norm": float(np.linalg.norm(origin_correction)),
        "maximum_direct_field_implementation_relative_defect": float(
            np.max(implementation_defect)
        ),
        "direct_field_evaluation_count": len(coordinates),
        "direct_field_wall_seconds": direct_wall,
        "state_dependent_coordinate_Jacobian_calls": 0,
        "new_continuous_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
    }
    arrays = {
        "training_online_coordinates": coordinates,
        "training_departure_coordinates": departures,
        "training_predicted_full_state_rates_per_second": predicted_full_rates,
        "training_exact_online_rates_per_second": exact_online_rates,
        "training_base_restricted_rates_per_second": base_online_rates,
        "training_direct_predicted_online_rates_per_second": predicted_online_rates,
        "q_rate_centers": departures,
        "q_rate_coefficients": coefficients,
        "q_rate_kernel_matrix": kernel,
        "q_rate_kernel_singular_values": np.linalg.svd(
            regularized, compute_uv=False
        ),
        "origin_q_rate_correction": origin_correction,
    }
    return arrays, metrics


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "online_field": {
            "state": "y470_equals_q162_z280_a28",
            "decoder": "independently_validated_compensated_decoder",
            "full_physical_rate": "independently_validated_shell_gated_degree45_field",
            "q162_rate": "frozen_anchor_restriction_plus_anchored_Gaussian_correction",
            "z280_rate": "fixed_memory_basis_transpose_times_full_rate",
            "a28_rate": "fixed_departure_basis_transpose_times_full_rate",
            "state_dependent_coordinate_Jacobian_online": False,
            "exact_rate_calls_online": 0,
        },
        "q_rate_closure": {
            "kernel": "anchor_subtracted_isotropic_Gaussian",
            "length_scale": Q_KERNEL_LENGTH_SCALE,
            "regularization": Q_KERNEL_REGULARIZATION,
            "training_exact_rate_count": 16,
            "correction_exactly_zero_at_anchor": True,
        },
        "binding_revealed_fit_gates": {
            "maximum_kernel_condition_number": 1.0e3,
            "maximum_training_full_coordinate_rate_relative_error": 5.0e-2,
            "maximum_training_q162_rate_relative_error": 1.0e-6,
            "maximum_training_z280_rate_relative_error": 1.5e-1,
            "maximum_training_a28_rate_relative_error": 1.5e-1,
            "maximum_origin_q_rate_correction_norm": 1.0e-14,
            "maximum_direct_field_implementation_relative_defect": 1.0e-12,
            "state_dependent_coordinate_Jacobian_calls_equal": 0,
        },
        "independent_exact_rate_holdout": {
            "source": "eight_geometry_only_mixed_corner_states_wp10c9d6c7c3b5c4f25cm",
            "count": PLANNED_RATE_EVALUATIONS,
            "coefficients_frozen_before_truth": True,
            "state_may_not_become_chart_center": True,
        },
        "binding_exact_truth_gates": {
            "completed_nonbase_rate_evaluations_equal": PLANNED_RATE_EVALUATIONS,
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
        },
        "binding_independent_field_gates": {
            "maximum_full_state_rate_relative_error": 0.15,
            "median_full_state_rate_relative_error": 0.075,
            "maximum_full_coordinate_rate_relative_error": 0.15,
            "median_full_coordinate_rate_relative_error": 0.075,
            "maximum_q162_rate_relative_error": 0.15,
            "median_q162_rate_relative_error": 0.075,
            "maximum_z280_rate_relative_error": 0.15,
            "maximum_a28_rate_relative_error": 0.15,
            "radial_sign_disagreement_count_equal": 0,
            "maximum_decoder_full_state_relative_error": 5.0e-3,
            "maximum_decoder_coordinate_relative_mismatch": 5.0e-3,
            "state_dependent_coordinate_Jacobian_calls_equal": 0,
        },
        "decision": {
            "pass_classification": "direct_470_coordinate_field_independently_validated",
            "fail_classification": "direct_470_coordinate_field_independent_validation_failed",
            "pass_authorizes_only": "definitions_only_one_recentered_transition_forecast_execution_manifest",
            "fail_authorizes_only": "definitions_only_direct_coordinate_field_revision_manifest",
        },
        "authorization_boundaries": {
            "new_truth_rate_calls_during_manifest": 0,
            "new_generator_assemblies": 0,
            "new_nonlinear_roots": 0,
            "propagated_states": 0,
            "trajectory_authorized": False,
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _fit_checks(metrics: dict, gates: dict) -> dict:
    return {
        "kernel_condition": metrics["kernel_condition_number"]
        <= gates["maximum_kernel_condition_number"],
        "full_coordinate_rate": metrics[
            "maximum_training_full_coordinate_rate_relative_error"
        ] <= gates["maximum_training_full_coordinate_rate_relative_error"],
        "q162_rate": metrics["maximum_training_q162_rate_relative_error"]
        <= gates["maximum_training_q162_rate_relative_error"],
        "z280_rate": metrics["maximum_training_z280_rate_relative_error"]
        <= gates["maximum_training_z280_rate_relative_error"],
        "a28_rate": metrics["maximum_training_a28_rate_relative_error"]
        <= gates["maximum_training_a28_rate_relative_error"],
        "origin": metrics["origin_q_rate_correction_norm"]
        <= gates["maximum_origin_q_rate_correction_norm"],
        "implementation": metrics[
            "maximum_direct_field_implementation_relative_defect"
        ] <= gates["maximum_direct_field_implementation_relative_defect"],
        "no_coordinate_Jacobian": metrics[
            "state_dependent_coordinate_Jacobian_calls"
        ] == gates["state_dependent_coordinate_Jacobian_calls_equal"],
        "rate_budget": metrics["new_continuous_rate_evaluations"] == 0,
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
            fieldnames=[
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ],
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
        raise RuntimeError("direct coordinate-field manifest already canonicalized")
    arrays, metrics = _fit_q_closure()
    contract = _contract()
    checks = _fit_checks(metrics, contract["binding_revealed_fit_gates"])
    if not all(checks.values()):
        raise RuntimeError(f"direct coordinate-field design failed: {checks}")
    holdout = _load_npz(HOLDOUT_GEOMETRY)
    if holdout["candidate_primitive_states"].shape != (8, 112, 5):
        raise RuntimeError("direct coordinate-field exact-rate holdout changed")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        CANONICAL_DIRECTORY / "direct_coordinate_field.npz", **arrays
    )
    _write_json(
        CANONICAL_DIRECTORY / "design_metrics.json",
        {"checks": checks, "fit": metrics},
    )
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "old_extension_sha256": _sha(OLD_EXTENSION),
            "decoder_repair_sha256": _sha(DECODER_REPAIR),
            "original_rate_arrays_sha256": _sha(ORIGINAL_RATE_ARRAYS),
            "mixed_rate_arrays_sha256": _sha(MIXED_RATE_ARRAYS),
            "holdout_geometry_sha256": _sha(HOLDOUT_GEOMETRY),
            "online_geometry_sha256": _sha(ONLINE_GEOMETRY),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "state_dependent_coordinate_Jacobian_online": False,
        "maximum_training_full_coordinate_rate_relative_error": metrics[
            "maximum_training_full_coordinate_rate_relative_error"
        ],
        "maximum_training_q162_rate_relative_error": metrics[
            "maximum_training_q162_rate_relative_error"
        ],
        "planned_independent_exact_rate_evaluations": PLANNED_RATE_EVALUATIONS,
        "coefficients_frozen_before_holdout_truth": True,
        "new_truth_rate_calls": 0,
        "new_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "trajectory_authorized": False,
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
        original_rate.THIS_RUNNER,
        original_rate.THIS_TEST,
        mixed_rate.THIS_RUNNER,
        mixed_rate.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
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
                "# Direct coordinate-field manifest WP10c9d6c7c3b5c4f25cn",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The online 470D field now uses a fixed restriction plus an anchor-subtracted Gaussian q162-rate correction. It performs no state-dependent coordinate-Jacobian construction. The z280 and a28 rates remain exact fixed projections of the independently validated full physical rate.",
                "",
                f"The revealed training maximum full-coordinate/q162 rate errors are `{metrics['maximum_training_full_coordinate_rate_relative_error']:.6e}` and `{metrics['maximum_training_q162_rate_relative_error']:.6e}`. The q correction is exactly zero at the chart anchor.",
                "",
                "Eight geometry-only mixed-corner states are frozen for fresh exact-rate validation. No state is propagated and no transition forecast, physical microburst, cycle evolution, or reduced slow evolution is authorized by this manifest.",
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
