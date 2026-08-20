#!/usr/bin/env python3
"""Blindly validate the frozen forward-quadratic authentic-center field."""

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

import run_causal_inner_forward_quadratic_field_revision_manifest_wp10c9d6c7c3b5c4f25cx as field_manifest  # noqa: E402
import run_causal_inner_forward_quadratic_geometry_preflight_wp10c9d6c7c3b5c4f25cy as manifest  # noqa: E402
import run_causal_inner_partitioned_authentic_center_field_blind_validation_wp10c9d6c7c3b5c4f25cw as exact_parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cz"
MANIFEST_COMMIT = "b5c770d8362934543e1af5cb5ffd0c4ba9307e9f"
MANIFEST_PARENT = "2c1c3545e37e939ef402c094a39f614550507342"
MANIFEST_TREE = "ad94e218cb72259bd5946022f7c3b62d9f0607ff"
EXACT_TRUTH_RUNNER_SHA256 = (
    "5a03d2cf98da8cd117eb0490e0045dd4b83a5a8bd5fd75e3747b7a1b7ba80e26"
)
EXACT_TRUTH_TEST_SHA256 = (
    "b33246536561cb47190ef7639133b7cbd640fe9e46319b89b97bb7f6b51b92ff"
)

PASS_CLASSIFICATION = (
    "forward_quadratic_authentic_center_field_independently_validated"
)
FAIL_CLASSIFICATION = (
    "forward_quadratic_authentic_center_field_blind_validation_failed"
)
PASS_AUTHORIZED_NEXT = "definitions_only_reduced_slow_atlas_integrator_manifest"
FAIL_AUTHORIZED_NEXT = "definitions_only_nonlinear_local_field_revision"

HOLDOUT_COUNT = field_manifest.NEW_GEOMETRY_COUNT
PHYSICAL_DIMENSION = field_manifest.PHYSICAL_DIMENSION
MEMORY_DIMENSION = field_manifest.MEMORY_DIMENSION
DEPARTURE_DIMENSION = field_manifest.DEPARTURE_DIMENSION

ARTIFACT = (
    "causal_inner_forward_quadratic_field_blind_validation_"
    "wp10c9d6c7c3b5c4f25cz"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_forward_quadratic_field_blind_validation_"
    "wp10c9d6c7c3b5c4f25cz.py"
)
THIS_TEST = (
    "tests/test_causal_inner_forward_quadratic_field_blind_validation_"
    "wp10c9d6c7c3b5c4f25cz.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_FORWARD_QUADRATIC_FIELD_"
    "BLIND_VALIDATION_WP10C9D6C7C3B5C4F25CZ_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PROGRESS_JSON = SCRATCH_DIRECTORY / "progress.json"
PROGRESS_NPZ = SCRATCH_DIRECTORY / "progress.npz"

GEOMETRY_ARRAYS = manifest.CANONICAL_DIRECTORY / "geometry_arrays.npz"
FROZEN_FIELD = (
    field_manifest.CANONICAL_DIRECTORY / "forward_quadratic_local_field.npz"
)

rate_engine = exact_parent.rate_engine
vector_field = exact_parent.vector_field
direct_manifest = exact_parent.direct_manifest

_plain = field_manifest._plain
_read = field_manifest._read
_write_json = field_manifest._write_json
_sha = field_manifest._sha
_checksums = field_manifest._checksums
_load_npz = field_manifest._load_npz
_append = exact_parent._append
_relative_error = exact_parent._relative_error


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


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("forward-quadratic geometry commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("forward-quadratic geometry lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("forward-quadratic geometry tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(manifest.CANONICAL_DIRECTORY / "geometry_metrics.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    input_contract = _read(
        manifest.CANONICAL_DIRECTORY / "input_execution_contract.json"
    )
    geometry = _load_npz(GEOMETRY_ARRAYS)
    field_hashes = _checksums(field_manifest.CANONICAL_DIRECTORY)
    field_summary = _read(field_manifest.CANONICAL_DIRECTORY / "summary.json")
    field_contract = _read(field_manifest.CANONICAL_DIRECTORY / "contract.json")
    field_provenance = _read(
        field_manifest.CANONICAL_DIRECTORY / "provenance.json"
    )
    closure = _load_npz(FROZEN_FIELD)
    if (
        not summary["passed"]
        or summary["classification"] != manifest.PASS_CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["completed_candidate_count"] != HOLDOUT_COUNT
        or summary["failed_candidate_count"] != 0
        or summary["directions_changed_after_manifest"]
        or summary["coefficients_changed_after_manifest"]
        or summary["new_exact_rate_calls"] != 0
        or not metrics["passed"]
        or not all(metrics["checks"].values())
        or not field_summary["passed"]
        or field_summary["classification"] != field_manifest.CLASSIFICATION
        or field_summary["authorized_next"] != manifest.WORK_PACKAGE
        or input_contract != field_contract
        or field_contract["decision"]["geometry_pass_authorizes_only"]
        != WORK_PACKAGE
        or field_contract["decision"]["blind_rate_pass_classification"]
        != PASS_CLASSIFICATION
        or field_contract["decision"]["blind_rate_fail_classification"]
        != FAIL_CLASSIFICATION
        or geometry["candidate_primitive_states"].shape != (HOLDOUT_COUNT, 112, 5)
        or geometry["candidate_local_coordinates"].shape != (HOLDOUT_COUNT, 470)
        or geometry["candidate_absolute_coordinates"].shape != (HOLDOUT_COUNT, 470)
        or closure["full_rate_forward_quadratic_coefficients"].shape != (5, 560)
        or closure["q162_Jacobian_affine_coefficients"].shape
        != (4, PHYSICAL_DIMENSION, 560)
        or not np.array_equal(
            geometry["candidate_directions"], closure["blind_directions"]
        )
        or not np.array_equal(
            geometry["candidate_component_bounds"],
            closure["blind_component_bounds"],
        )
    ):
        raise RuntimeError("forward-quadratic blind-rate authorization changed")
    for source, expected in provenance["source_hashes"].items():
        if _sha(ROOT / source) != expected:
            raise RuntimeError(f"forward-quadratic geometry source changed: {source}")
    for source, expected in field_provenance["source_hashes"].items():
        if _sha(ROOT / source) != expected:
            raise RuntimeError(f"forward-quadratic field source changed: {source}")
    if (
        _sha(GEOMETRY_ARRAYS) != hashes["geometry_arrays.npz"]
        or _sha(FROZEN_FIELD) != field_hashes["forward_quadratic_local_field.npz"]
    ):
        raise RuntimeError("forward-quadratic blind-rate input changed")
    for name, expected in field_manifest.training._thread_environment().items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("forward-quadratic blind validation requires a clean tracked tree")
    return {
        "summary": summary,
        "metrics": metrics,
        "hashes": hashes,
        "geometry": geometry,
        "contract": field_contract,
        "field_hashes": field_hashes,
        "closure": closure,
    }


def _load_inputs(frozen: dict) -> dict:
    geometry = frozen["geometry"]
    model = vector_field.ReducedVectorField()
    direct = direct_manifest.DirectCoordinateField(
        _load_npz(field_manifest.partition.DIRECT_FIELD), model=model
    )
    field = field_manifest.ForwardQuadraticAuthenticCenterField(
        frozen["closure"], model=model, direct=direct
    )
    states = np.asarray(geometry["candidate_primitive_states"], dtype=float)
    local = np.asarray(geometry["candidate_local_coordinates"], dtype=float)
    absolute = np.asarray(geometry["candidate_absolute_coordinates"], dtype=float)
    reconstructed = []
    factors = []
    for state in states:
        coordinate, reconstruction = model.coordinate(state)
        reconstructed.append(coordinate)
        factors.append(np.asarray(reconstruction, dtype=float))
    reconstructed = np.asarray(reconstructed)
    coordinate_errors = np.asarray(
        [
            _relative_error(left, right)
            for left, right in zip(reconstructed, absolute)
        ]
    )
    weights = np.asarray([field.weight(value) for value in local])
    if (
        states.shape != (HOLDOUT_COUNT, 112, 5)
        or local.shape != (HOLDOUT_COUNT, 470)
        or absolute.shape != (HOLDOUT_COUNT, 470)
        or float(np.max(coordinate_errors)) > 1.0e-8
        or not np.array_equal(weights, np.ones(HOLDOUT_COUNT))
    ):
        raise RuntimeError("forward-quadratic blind inputs changed")
    return {
        "model": model,
        "direct": direct,
        "field": field,
        "states": states,
        "local_coordinates": local,
        "absolute_coordinates": absolute,
        "exact_scaled_deltas": np.asarray(
            geometry["candidate_absolute_scaled_deltas"], dtype=float
        ),
        "coordinate_roundtrip_relative_errors": coordinate_errors,
        "minimum_input_reconstruction_factor": float(
            min(np.min(value) for value in factors)
        ),
        "partition_weights": weights,
        "labels": tuple(f"quadratic_holdout_{index}" for index in range(HOLDOUT_COUNT)),
    }


def _online_prediction_without_coordinate_jacobian(
    inputs: dict, index: int
) -> tuple[dict[str, np.ndarray], float]:
    chart_tools = vector_field.manifest.parent.geometry.chart_tools
    original = chart_tools._coordinate_jacobian

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("online forward-quadratic field attempted a coordinate Jacobian")

    chart_tools._coordinate_jacobian = forbidden
    began = time.perf_counter()
    try:
        local = inputs["local_coordinates"][index]
        full = inputs["field"].full_state_rate(local)
        coordinate = inputs["field"].field(local)
        decoded_delta = inputs["field"].decoded_delta(local)
        decoded_state = inputs["field"].decoded_state(local)
        q_jacobian = inputs["field"].q162_jacobian(local)
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
    return exact_parent._progress_array_shapes()


def _progress_identity() -> dict:
    return {
        "work_package": WORK_PACKAGE,
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_hashes": _checksums(manifest.CANONICAL_DIRECTORY),
        "field_hashes": _checksums(field_manifest.CANONICAL_DIRECTORY),
        # These hashes identify the source that executed the four truth calls.
        # A post-truth patch only repaired gate lookup during canonicalization.
        "runner_sha256": EXACT_TRUTH_RUNNER_SHA256,
        "test_sha256": EXACT_TRUTH_TEST_SHA256,
        "geometry_arrays_sha256": _sha(GEOMETRY_ARRAYS),
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
        raise RuntimeError("forward-quadratic validation checkpoint is incomplete")
    recorded = _read(PROGRESS_JSON)
    if recorded["identity"] != _progress_identity():
        raise RuntimeError("forward-quadratic validation checkpoint identity changed")
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
        raise RuntimeError("forward-quadratic checkpoint dimensions changed")
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
    decoded_physical = vector_field.manifest.parent.geometry.chart_tools._state_audit(
        model.components["context"], predicted["decoded_state"]
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
    data = rate_engine.manifest.prior_screen.manifest.parent.manifest.failed_screen._anchor_data(
        "primary"
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
        "maximum_raw_Schur_condition_number": maximum("raw_Schur_condition_number"),
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
        )
        if evaluations
        else math.inf,
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
    return exact_parent._checks(metrics, gates)


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
        raise RuntimeError("forward-quadratic blind validation already canonicalized")
    inputs = _load_inputs(frozen)
    metrics, arrays = _execute(inputs)
    gates = dict(frozen["contract"]["blind_rate_validation"])
    geometry_gates = frozen["contract"]["geometry_preflight"]
    gates["maximum_decoder_relative_error"] = geometry_gates[
        "maximum_decoder_relative_error"
    ]
    gates["maximum_decoder_coordinate_relative_mismatch"] = geometry_gates[
        "maximum_decoder_coordinate_relative_mismatch"
    ]
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
        "maximum_decoder_relative_error": metrics["maximum_decoder_relative_error"],
        "coefficients_refit_after_holdout_truth": False,
        "postprocessing_gate_lookup_repaired_after_truth": True,
        "truth_calls_repeated_after_postprocessing_repair": 0,
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
        CANONICAL_DIRECTORY / "input_execution_contract.json", frozen["contract"]
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "geometry_hashes": frozen["hashes"],
            "field_hashes": frozen["field_hashes"],
            "geometry_arrays_sha256": _sha(GEOMETRY_ARRAYS),
            "frozen_field_sha256": _sha(FROZEN_FIELD),
            "coefficients_refit_after_holdout_truth": False,
            "exact_truth_runner_sha256": EXACT_TRUTH_RUNNER_SHA256,
            "exact_truth_test_sha256": EXACT_TRUTH_TEST_SHA256,
            "postprocessing_gate_lookup_repaired_after_truth": True,
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        field_manifest.THIS_RUNNER,
        field_manifest.THIS_TEST,
        exact_parent.THIS_RUNNER,
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
            "exact_truth_runner_sha256": EXACT_TRUTH_RUNNER_SHA256,
            "exact_truth_test_sha256": EXACT_TRUTH_TEST_SHA256,
            "postprocessing_only_recovery": True,
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
                for name in field_manifest.training._thread_environment()
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
                "# Forward-quadratic authentic-center field blind validation WP10c9d6c7c3b5c4f25cz",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{metrics['completed_exact_rate_calls']}` of `{HOLDOUT_COUNT}` frozen blind exact-rate evaluations with `{metrics['failed_exact_rate_calls']}` failures.",
                "",
                f"Maximum full-state/full-coordinate/q162/z280/a28 errors are `{metrics['maximum_full_state_rate_relative_error']:.6e}`, `{metrics['maximum_full_coordinate_rate_relative_error']:.6e}`, `{metrics['maximum_q162_rate_relative_error']:.6e}`, `{metrics['maximum_z280_rate_relative_error']:.6e}`, and `{metrics['maximum_a28_rate_relative_error']:.6e}`.",
                "",
                f"Maximum transported q162-Jacobian error is `{metrics['maximum_q162_Jacobian_relative_error']:.6e}`. Maximum decoder error is `{metrics['maximum_decoder_relative_error']:.6e}`. Median online evaluation took `{metrics['median_online_field_wall_seconds']:.6e}` seconds with no state-dependent coordinate-Jacobian assembly.",
                "",
                "The coefficients were not changed after blind truth. No state was propagated.",
                "",
                "All four truth calls completed before canonicalization exposed a gate-location lookup bug. Canonicalization reused their hash-locked checkpoint and merged the decoder limits already frozen under the geometry subsection; no truth call was repeated.",
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
