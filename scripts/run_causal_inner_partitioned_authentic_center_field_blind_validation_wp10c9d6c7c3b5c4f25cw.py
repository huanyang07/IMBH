#!/usr/bin/env python3
"""Blindly validate the frozen partitioned authentic-center field."""

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
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_partitioned_authentic_center_field_revision_manifest_wp10c9d6c7c3b5c4f25cv as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cw"
MANIFEST_COMMIT = "2037789867e02e38427f56d71228cade7e36f6fd"
MANIFEST_PARENT = "1cd3398bd1fb6338156cefac0f7792e0071ad244"
MANIFEST_TREE = "ded3dd0d74e682c8454d4a0334f7a15483a3484d"

PASS_CLASSIFICATION = "partitioned_authentic_center_field_independently_validated"
FAIL_CLASSIFICATION = "partitioned_authentic_center_field_blind_validation_failed"
PASS_AUTHORIZED_NEXT = "definitions_only_reduced_slow_atlas_integrator_manifest"
FAIL_AUTHORIZED_NEXT = "definitions_only_partitioned_field_revision"

HOLDOUT_COUNT = manifest.HOLDOUT_COUNT
PHYSICAL_DIMENSION = manifest.PHYSICAL_DIMENSION
MEMORY_DIMENSION = manifest.MEMORY_DIMENSION
DEPARTURE_DIMENSION = manifest.DEPARTURE_DIMENSION

ARTIFACT = (
    "causal_inner_partitioned_authentic_center_field_blind_validation_"
    "wp10c9d6c7c3b5c4f25cw"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_partitioned_authentic_center_field_blind_validation_"
    "wp10c9d6c7c3b5c4f25cw.py"
)
THIS_TEST = (
    "tests/test_causal_inner_partitioned_authentic_center_field_blind_validation_"
    "wp10c9d6c7c3b5c4f25cw.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PARTITIONED_AUTHENTIC_CENTER_"
    "FIELD_BLIND_VALIDATION_WP10C9D6C7C3B5C4F25CW_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PROGRESS_JSON = SCRATCH_DIRECTORY / "progress.json"
PROGRESS_NPZ = SCRATCH_DIRECTORY / "progress.npz"

FROZEN_FIELD = manifest.CANONICAL_DIRECTORY / "partitioned_local_field.npz"

parent = manifest.parent
rate_engine = parent.rate_engine
vector_field = parent.vector_field
direct_manifest = parent.direct_manifest

_plain = manifest._plain
_read = manifest._read
_write_json = manifest._write_json
_sha = manifest._sha
_checksums = manifest._checksums
_load_npz = manifest._load_npz


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


def _append(array: np.ndarray, value: np.ndarray) -> np.ndarray:
    item = np.asarray(value, dtype=float)
    return np.concatenate((array, item.reshape((1,) + item.shape)), axis=0)


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))
        / max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    )


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("partitioned-field manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("partitioned-field manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("partitioned-field manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    metrics = _read(manifest.CANONICAL_DIRECTORY / "design_metrics.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    lock = _read(manifest.CANONICAL_DIRECTORY / "parent_lock.json")
    closure = _load_npz(FROZEN_FIELD)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["coefficients_frozen_before_blind_holdout_truth"]
        or summary["new_exact_rate_calls"] != 0
        or summary["blind_holdout_rate_calls"] != 0
        or not metrics["passed"]
        or not all(metrics["checks"].values())
        or contract["blind_holdout_execution"]["work_package"] != WORK_PACKAGE
        or contract["blind_holdout_execution"]["count"] != HOLDOUT_COUNT
        or not contract["blind_holdout_execution"]["coefficients_may_not_change"]
        or contract["decision"]["pass_classification"] != PASS_CLASSIFICATION
        or contract["decision"]["fail_classification"] != FAIL_CLASSIFICATION
        or closure["holdout_primitive_states"].shape != (HOLDOUT_COUNT, 112, 5)
        or closure["holdout_local_coordinates"].shape != (HOLDOUT_COUNT, 470)
        or closure["holdout_absolute_coordinates"].shape != (HOLDOUT_COUNT, 470)
        or closure["q162_Jacobian_affine_coefficients"].shape
        != (4, PHYSICAL_DIMENSION, 560)
        or not np.array_equal(
            closure["geometry_partition_weights"][-HOLDOUT_COUNT:],
            np.ones(HOLDOUT_COUNT),
        )
    ):
        raise RuntimeError("blind partitioned-field validation contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"partitioned-field source changed: {relative}")
    if (
        _sha(FROZEN_FIELD) != hashes["partitioned_local_field.npz"]
        or _sha(FROZEN_FIELD)
        != _checksums(manifest.CANONICAL_DIRECTORY)["partitioned_local_field.npz"]
        or _sha(manifest.PARENT_TRUTH) != lock["training_truth_sha256"]
    ):
        raise RuntimeError("blind partitioned-field input changed")
    for name, expected in parent._thread_environment().items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("blind partitioned-field validation requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "metrics": metrics,
        "hashes": hashes,
        "closure": closure,
    }


def _load_inputs(frozen: dict) -> dict:
    closure = frozen["closure"]
    model = vector_field.ReducedVectorField()
    direct = direct_manifest.DirectCoordinateField(
        _load_npz(manifest.DIRECT_FIELD), model=model
    )
    field = manifest.PartitionedAuthenticCenterField(
        closure, model=model, direct=direct
    )
    states = np.asarray(closure["holdout_primitive_states"], dtype=float)
    local = np.asarray(closure["holdout_local_coordinates"], dtype=float)
    absolute = np.asarray(closure["holdout_absolute_coordinates"], dtype=float)
    reconstructed = []
    factors = []
    for state in states:
        coordinate, reconstruction = model.coordinate(state)
        reconstructed.append(coordinate)
        factors.append(np.asarray(reconstruction, dtype=float))
    reconstructed = np.asarray(reconstructed)
    coordinate_errors = np.asarray(
        [_relative_error(left, right) for left, right in zip(reconstructed, absolute)]
    )
    weights = np.asarray([field.weight(value) for value in local])
    if (
        states.shape != (HOLDOUT_COUNT, 112, 5)
        or local.shape != (HOLDOUT_COUNT, 470)
        or absolute.shape != (HOLDOUT_COUNT, 470)
        or float(np.max(coordinate_errors)) > 1.0e-8
        or not np.array_equal(weights, np.ones(HOLDOUT_COUNT))
    ):
        raise RuntimeError("blind holdout inputs changed")
    return {
        "model": model,
        "direct": direct,
        "field": field,
        "states": states,
        "local_coordinates": local,
        "absolute_coordinates": absolute,
        "exact_scaled_deltas": np.asarray(
            [
                ((state - model.base_state) / model.columns).ravel()
                for state in states
            ]
        ),
        "coordinate_roundtrip_relative_errors": coordinate_errors,
        "minimum_input_reconstruction_factor": float(
            min(np.min(value) for value in factors)
        ),
        "partition_weights": weights,
        "labels": ("holdout_0", "holdout_1", "holdout_2", "holdout_3"),
    }


def _online_prediction_without_coordinate_jacobian(
    inputs: dict, index: int
) -> tuple[dict[str, np.ndarray], float]:
    chart_tools = vector_field.manifest.parent.geometry.chart_tools
    original = chart_tools._coordinate_jacobian

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("online partitioned field attempted a coordinate-Jacobian build")

    chart_tools._coordinate_jacobian = forbidden
    began = time.perf_counter()
    try:
        local = inputs["local_coordinates"][index]
        full = inputs["field"].full_state_rate(local)
        coordinate = inputs["field"].field(local)
        decoded_delta = inputs["field"].decoded_delta(local)
        decoded_state = inputs["field"].decoded_state(local)
        features = inputs["field"]._features(local)
        q_jacobian = np.einsum(
            "f,fij->ij",
            features,
            inputs["field"].q_jacobian_coefficients,
        )
    finally:
        wall = time.perf_counter() - began
        chart_tools._coordinate_jacobian = original
    return {
        "full_rate": full,
        "coordinate_rate": coordinate,
        "decoded_delta": decoded_delta,
        "decoded_state": decoded_state,
        "q162_Jacobian": q_jacobian,
    }, wall


def _progress_array_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "total_rates_per_second": (560,),
        "free_rates_per_second": (560,),
        "physical_reaction_actions_per_second": (560,),
        "multiplier_coordinates_per_second": (3,),
        "exact_coordinate_rates_per_second": (470,),
        "predicted_full_rates_per_second": (560,),
        "predicted_coordinate_rates_per_second": (470,),
        "exact_q162_Jacobians": (PHYSICAL_DIMENSION, 560),
        "predicted_q162_Jacobians": (PHYSICAL_DIMENSION, 560),
        "decoded_scaled_deltas": (560,),
        "decoded_absolute_coordinates": (470,),
    }


def _progress_identity() -> dict:
    return {
        "work_package": WORK_PACKAGE,
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_hashes": _checksums(manifest.CANONICAL_DIRECTORY),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "frozen_field_sha256": _sha(FROZEN_FIELD),
    }


def _empty_progress() -> dict:
    progress = {"identity": _progress_identity(), "evaluations": [], "failures": []}
    for name, shape in _progress_array_shapes().items():
        progress[name] = np.empty((0,) + shape, dtype=float)
    return progress


def _save_progress(progress: dict) -> None:
    SCRATCH_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        PROGRESS_JSON,
        {
            "identity": progress["identity"],
            "evaluations": progress["evaluations"],
            "failures": progress["failures"],
        },
    )
    _write_npz(
        PROGRESS_NPZ,
        {name: progress[name] for name in _progress_array_shapes()},
    )


def _load_or_create_progress() -> dict:
    if not PROGRESS_JSON.exists() and not PROGRESS_NPZ.exists():
        return _empty_progress()
    if not PROGRESS_JSON.exists() or not PROGRESS_NPZ.exists():
        raise RuntimeError("blind field-validation checkpoint is incomplete")
    recorded = _read(PROGRESS_JSON)
    if recorded["identity"] != _progress_identity():
        raise RuntimeError("blind field-validation checkpoint identity changed")
    progress = {
        "identity": recorded["identity"],
        "evaluations": recorded["evaluations"],
        "failures": recorded["failures"],
        **_load_npz(PROGRESS_NPZ),
    }
    count = len(progress["evaluations"])
    if (
        count > HOLDOUT_COUNT
        or [item["candidate_index"] for item in progress["evaluations"]]
        != list(range(count))
        or any(
            progress[name].shape != (count,) + shape
            for name, shape in _progress_array_shapes().items()
        )
    ):
        raise RuntimeError("blind field-validation checkpoint dimensions changed")
    return progress


def _evaluate_one(inputs: dict, progress: dict, index: int, data: dict) -> None:
    state = inputs["states"][index]
    model = inputs["model"]
    predicted, online_wall = _online_prediction_without_coordinate_jacobian(
        inputs, index
    )
    item, arrays = rate_engine.manifest.prior_screen._continuous_rate(data, state)
    exact_jacobian, jacobian_metrics = (
        vector_field.manifest.parent.geometry.chart_tools._coordinate_jacobian(
            state, model.components
        )
    )
    total_rate = np.asarray(arrays["total_rate"], dtype=float)
    exact_coordinate = np.concatenate(
        (
            exact_jacobian @ total_rate,
            model.memory_basis.T @ total_rate,
            model.departure_basis.T @ total_rate,
        )
    )
    decoded_coordinate, decoded_factors = model.coordinate(predicted["decoded_state"])
    decoded_physical = (
        vector_field.manifest.parent.geometry.chart_tools._state_audit(
            model.components["context"], predicted["decoded_state"]
        )
    )
    q_slice = slice(0, PHYSICAL_DIMENSION)
    z_slice = slice(PHYSICAL_DIMENSION, PHYSICAL_DIMENSION + MEMORY_DIMENSION)
    a_slice = slice(-DEPARTURE_DIMENSION, None)
    item.update(
        {
            "candidate_index": index,
            "candidate_label": inputs["labels"][index],
            "partition_weight": float(inputs["partition_weights"][index]),
            "center_local_coordinate_load": float(
                np.max(np.abs(inputs["local_coordinates"][index]))
            ),
            "full_state_rate_relative_error": _relative_error(
                predicted["full_rate"], total_rate
            ),
            "full_coordinate_rate_relative_error": _relative_error(
                predicted["coordinate_rate"], exact_coordinate
            ),
            "q162_rate_relative_error": _relative_error(
                predicted["coordinate_rate"][q_slice], exact_coordinate[q_slice]
            ),
            "z280_rate_relative_error": _relative_error(
                predicted["coordinate_rate"][z_slice], exact_coordinate[z_slice]
            ),
            "a28_rate_relative_error": _relative_error(
                predicted["coordinate_rate"][a_slice], exact_coordinate[a_slice]
            ),
            "q162_Jacobian_relative_error": _relative_error(
                predicted["q162_Jacobian"], exact_jacobian
            ),
            "decoder_relative_error": _relative_error(
                predicted["decoded_delta"], inputs["exact_scaled_deltas"][index]
            ),
            "decoder_coordinate_relative_mismatch": _relative_error(
                decoded_coordinate, inputs["absolute_coordinates"][index]
            ),
            "coordinate_Jacobian_rank": jacobian_metrics["rank"],
            "coordinate_Jacobian_condition_number": jacobian_metrics[
                "condition_number"
            ],
            "online_field_wall_seconds": online_wall,
            "online_state_dependent_coordinate_Jacobian_calls": 0,
            "offline_truth_coordinate_Jacobian_calls": 1,
            "decoded_minimum_reconstruction_factor": min(
                float(np.min(decoded_factors)),
                decoded_physical["minimum_reconstruction_factor"],
            ),
            "decoded_maximum_H_over_R": decoded_physical["maximum_h_over_r"],
            "decoded_minimum_scattering_optical_depth": decoded_physical[
                "minimum_scattering_optical_depth"
            ],
        }
    )
    progress["evaluations"].append(item)
    values = {
        "total_rates_per_second": total_rate,
        "free_rates_per_second": arrays["free_rate"],
        "physical_reaction_actions_per_second": arrays["reaction_action"],
        "multiplier_coordinates_per_second": arrays["multiplier"],
        "exact_coordinate_rates_per_second": exact_coordinate,
        "predicted_full_rates_per_second": predicted["full_rate"],
        "predicted_coordinate_rates_per_second": predicted["coordinate_rate"],
        "exact_q162_Jacobians": exact_jacobian,
        "predicted_q162_Jacobians": predicted["q162_Jacobian"],
        "decoded_scaled_deltas": predicted["decoded_delta"],
        "decoded_absolute_coordinates": decoded_coordinate,
    }
    for name, value in values.items():
        progress[name] = _append(progress[name], value)


def _execute(inputs: dict) -> tuple[dict, dict[str, np.ndarray]]:
    progress = _load_or_create_progress()
    resumed = len(progress["evaluations"])
    data = (
        rate_engine.manifest.prior_screen.manifest.parent.manifest.failed_screen._anchor_data(
            "primary"
        )
    )
    began = time.perf_counter()
    for index in range(resumed, HOLDOUT_COUNT):
        try:
            _evaluate_one(inputs, progress, index, data)
            status = "accepted"
        except Exception as error:
            progress["failures"].append(
                {
                    "candidate_index": index,
                    "candidate_label": inputs["labels"][index],
                    "reason": type(error).__name__,
                    "message": str(error),
                }
            )
            status = "failed"
        _save_progress(progress)
        print(
            json.dumps(
                {
                    "blind_exact_rate_evaluation": index + 1,
                    "total": HOLDOUT_COUNT,
                    "candidate": inputs["labels"][index],
                    "status": status,
                    "elapsed_this_process_seconds": time.perf_counter() - began,
                }
            ),
            flush=True,
        )
        if progress["failures"]:
            break
    count = len(progress["evaluations"])
    arrays = {
        "holdout_primitive_states": inputs["states"][:count],
        "holdout_local_coordinates": inputs["local_coordinates"][:count],
        "holdout_absolute_coordinates": inputs["absolute_coordinates"][:count],
        **{name: progress[name] for name in _progress_array_shapes()},
    }
    evaluations = progress["evaluations"]

    def values(name: str) -> list[float]:
        return [float(item[name]) for item in evaluations]

    def maximum(name: str, default=math.inf) -> float:
        entries = values(name)
        return max(entries) if entries else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        entries = values(name)
        return min(entries) if entries else float(default)

    metrics = {
        "planned_exact_rate_calls": HOLDOUT_COUNT,
        "completed_exact_rate_calls": count,
        "failed_exact_rate_calls": len(progress["failures"]),
        "failures": progress["failures"],
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
        "maximum_incoming_excision_characteristics": int(
            maximum("incoming_excision_characteristics", 1.0e9)
        ),
        "maximum_full_state_rate_relative_error": maximum(
            "full_state_rate_relative_error"
        ),
        "maximum_full_coordinate_rate_relative_error": maximum(
            "full_coordinate_rate_relative_error"
        ),
        "maximum_q162_rate_relative_error": maximum("q162_rate_relative_error"),
        "maximum_z280_rate_relative_error": maximum("z280_rate_relative_error"),
        "maximum_a28_rate_relative_error": maximum("a28_rate_relative_error"),
        "maximum_q162_Jacobian_relative_error": maximum(
            "q162_Jacobian_relative_error"
        ),
        "maximum_decoder_relative_error": maximum("decoder_relative_error"),
        "maximum_decoder_coordinate_relative_mismatch": maximum(
            "decoder_coordinate_relative_mismatch"
        ),
        "minimum_partition_weight": minimum("partition_weight"),
        "minimum_decoded_reconstruction_factor": minimum(
            "decoded_minimum_reconstruction_factor", math.inf
        ),
        "maximum_decoded_H_over_R": maximum("decoded_maximum_H_over_R"),
        "minimum_decoded_scattering_optical_depth": minimum(
            "decoded_minimum_scattering_optical_depth"
        ),
        "median_online_field_wall_seconds": float(
            np.median(values("online_field_wall_seconds"))
        ) if evaluations else math.inf,
        "maximum_online_field_wall_seconds": maximum("online_field_wall_seconds"),
        "online_state_dependent_coordinate_Jacobian_calls": int(
            sum(values("online_state_dependent_coordinate_Jacobian_calls"))
        ),
        "offline_truth_coordinate_Jacobian_calls": count,
        "new_exact_continuous_rate_calls": count - resumed,
        "resumed_exact_rate_calls": resumed,
        "total_available_exact_rate_calls": count,
        "coefficients_refit_after_holdout_truth": False,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "wall_seconds_this_process": time.perf_counter() - began,
        "evaluations": evaluations,
    }
    return metrics, arrays


def _checks(metrics: dict, gates: dict) -> dict:
    return {
        "completed": metrics["completed_exact_rate_calls"]
        == gates["completed_exact_rate_calls_equal"],
        "failed": metrics["failed_exact_rate_calls"]
        == gates["failed_exact_rate_calls_equal"],
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
        ] <= gates["maximum_coordinate_Jacobian_condition_number"],
        "height": metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"],
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"],
        "incoming_excision": metrics["maximum_incoming_excision_characteristics"]
        == gates["maximum_incoming_excision_characteristics_equal"],
        "full_state": metrics["maximum_full_state_rate_relative_error"]
        <= gates["maximum_full_state_rate_relative_error"],
        "full_coordinate": metrics["maximum_full_coordinate_rate_relative_error"]
        <= gates["maximum_full_coordinate_rate_relative_error"],
        "q162": metrics["maximum_q162_rate_relative_error"]
        <= gates["maximum_q162_rate_relative_error"],
        "z280": metrics["maximum_z280_rate_relative_error"]
        <= gates["maximum_z280_rate_relative_error"],
        "a28": metrics["maximum_a28_rate_relative_error"]
        <= gates["maximum_a28_rate_relative_error"],
        "q162_Jacobian": metrics["maximum_q162_Jacobian_relative_error"]
        <= gates["maximum_q162_Jacobian_relative_error"],
        "decoder": metrics["maximum_decoder_relative_error"]
        <= gates["maximum_decoder_relative_error"],
        "decoder_coordinate": metrics[
            "maximum_decoder_coordinate_relative_mismatch"
        ] <= gates["maximum_decoder_coordinate_relative_mismatch"],
        "partition": metrics["minimum_partition_weight"] == 1.0,
        "no_online_coordinate_Jacobian": metrics[
            "online_state_dependent_coordinate_Jacobian_calls"
        ] == gates["online_state_dependent_coordinate_Jacobian_calls_equal"],
        "generator_budget": metrics["new_complete_generator_assemblies"]
        == gates["new_complete_generator_assemblies_equal"],
        "root_budget": metrics["new_nonlinear_fixed_Q_roots"]
        == gates["new_nonlinear_fixed_Q_roots_equal"],
        "propagation_budget": metrics["propagated_states"]
        == gates["propagated_states_equal"],
        "coefficient_blindness": not metrics["coefficients_refit_after_holdout_truth"],
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
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("partitioned-field blind validation already canonicalized")
    inputs = _load_inputs(frozen)
    metrics, arrays = _execute(inputs)
    gates = frozen["contract"]["blind_holdout_execution"]
    checks = _checks(metrics, gates)
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = PASS_AUTHORIZED_NEXT if passed else FAIL_AUTHORIZED_NEXT
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "completed_exact_rate_calls": metrics["completed_exact_rate_calls"],
        "failed_exact_rate_calls": metrics["failed_exact_rate_calls"],
        "blind_holdout_passed": passed,
        "maximum_full_state_rate_relative_error": metrics[
            "maximum_full_state_rate_relative_error"
        ],
        "maximum_full_coordinate_rate_relative_error": metrics[
            "maximum_full_coordinate_rate_relative_error"
        ],
        "maximum_q162_rate_relative_error": metrics[
            "maximum_q162_rate_relative_error"
        ],
        "maximum_q162_Jacobian_relative_error": metrics[
            "maximum_q162_Jacobian_relative_error"
        ],
        "maximum_decoder_relative_error": metrics[
            "maximum_decoder_relative_error"
        ],
        "coefficients_refit_after_holdout_truth": False,
        "online_state_dependent_coordinate_Jacobian_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "validation_metrics.json",
        {"checks": checks, "passed": passed, **metrics},
    )
    _write_npz(CANONICAL_DIRECTORY / "validation_arrays.npz", arrays)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "input_execution_contract.json",
        frozen["contract"],
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
            "partitioned_field_sha256": _sha(FROZEN_FIELD),
            "coefficients_refit_after_holdout_truth": False,
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        rate_engine.manifest.prior_screen.THIS_RUNNER,
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
            "resumed_from_exact_rate_count": metrics["resumed_exact_rate_calls"],
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
                "# Partitioned authentic-center field blind validation WP10c9d6c7c3b5c4f25cw",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{metrics['completed_exact_rate_calls']}` of `{HOLDOUT_COUNT}` frozen blind exact-rate evaluations with `{metrics['failed_exact_rate_calls']}` failures.",
                "",
                f"Maximum full-state/full-coordinate/q162/z280/a28 errors are `{metrics['maximum_full_state_rate_relative_error']:.6e}`, `{metrics['maximum_full_coordinate_rate_relative_error']:.6e}`, `{metrics['maximum_q162_rate_relative_error']:.6e}`, `{metrics['maximum_z280_rate_relative_error']:.6e}`, and `{metrics['maximum_a28_rate_relative_error']:.6e}`.",
                "",
                f"Maximum transported q162-Jacobian error is `{metrics['maximum_q162_Jacobian_relative_error']:.6e}`. Maximum decoder error is `{metrics['maximum_decoder_relative_error']:.6e}`. The median online field/decoder evaluation took `{metrics['median_online_field_wall_seconds']:.6e}` seconds and assembled no state-dependent coordinate Jacobian.",
                "",
                "The frozen coefficients were not changed after holdout truth. No state was propagated.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No physical microburst, predictive cycle, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    _run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
