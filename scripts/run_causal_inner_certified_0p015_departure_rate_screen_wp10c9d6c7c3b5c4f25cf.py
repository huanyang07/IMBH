#!/usr/bin/env python3
"""Execute the frozen targeted exact-rate screen at departure bound 0.015."""

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

import run_causal_inner_certified_0p015_departure_rate_design_manifest_wp10c9d6c7c3b5c4f25ce as manifest  # noqa: E402
import run_causal_inner_expanded_departure_rate_screen_wp10c9d6c7c3b5c4f25be as rate_engine  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cf"
MANIFEST_COMMIT = "9b01876c659ba7ab94b2868d9c5d5f73145a5b9e"
MANIFEST_PARENT = "b8c461d852944c3911a546ff028041e7c886d509"
MANIFEST_TREE = "7970ec2ec3b48b7ddc5368beddb3d8a83ff4665c"

FAIL_CLASSIFICATION = "targeted_0p015_exact_rate_screen_failed"
OUTWARD_CLASSIFICATION = (
    "certified_0p015_exact_forward_chart_exit_recentered_atlas_required"
)
INWARD_CLASSIFICATION = (
    "certified_0p015_exact_forward_inward_turn_saturation_candidate"
)
TANGENTIAL_CLASSIFICATION = "certified_0p015_exact_forward_turn_unresolved"

ARTIFACT = (
    "causal_inner_certified_0p015_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25cf"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_certified_0p015_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25cf.py"
)
THIS_TEST = (
    "tests/test_causal_inner_certified_0p015_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25cf.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CERTIFIED_0P015_DEPARTURE_"
    "RATE_SCREEN_WP10C9D6C7C3B5C4F25CF_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PROGRESS_JSON = SCRATCH_DIRECTORY / "progress.json"
PROGRESS_NPZ = SCRATCH_DIRECTORY / "progress.npz"

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


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(expected))
        / max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    )


def _append(array: np.ndarray, value: np.ndarray) -> np.ndarray:
    item = np.asarray(value, dtype=float)
    return np.concatenate((array, item.reshape((1,) + item.shape)), axis=0)


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("targeted rate manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("targeted rate manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("targeted rate manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    design_json = _read(manifest.CANONICAL_DIRECTORY / "rate_design.json")
    design = _load_npz(manifest.CANONICAL_DIRECTORY / "rate_design.npz")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_exact_rate_evaluations"]
        != manifest.PLANNED_RATE_EVALUATIONS
        or summary["new_truth_calls"] != 0
        or summary["trajectory_authorized"]
        or summary["predictive_cycle_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or tuple(design["parent_candidate_indices"])
        != manifest.SELECTED_CANDIDATE_INDICES
        or tuple(design_json["direction_labels"])
        != manifest.TARGET_DIRECTION_LABELS
    ):
        raise RuntimeError("targeted exact-rate contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"targeted rate manifest source changed: {relative}")
    lock = _read(manifest.CANONICAL_DIRECTORY / "parent_lock.json")
    locked_paths = {
        "expanded_chart_states_sha256": manifest.PARENT_ARRAYS,
        "old_coefficients_sha256": manifest.OLD_COEFFICIENTS,
        "online_geometry_sha256": manifest.ONLINE_GEOMETRY,
        "generator_sha256": manifest.GENERATOR,
    }
    for field, path in locked_paths.items():
        if _sha(path) != lock[field]:
            raise RuntimeError(f"targeted rate input changed: {path}")
    for name, expected in manifest.parent.manifest.parent.vector_field.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("targeted exact-rate screen requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "design": design,
        "design_json": design_json,
        "hashes": hashes,
    }


def _load_inputs(frozen: dict) -> dict:
    parent_arrays = _load_npz(manifest.PARENT_ARRAYS)
    geometry = _load_npz(manifest.ONLINE_GEOMETRY)
    generator = _load_npz(manifest.GENERATOR)
    coefficients = _load_npz(manifest.OLD_COEFFICIENTS)
    indices = np.asarray(frozen["design"]["parent_candidate_indices"], dtype=int)
    states = parent_arrays["candidate_primitive_states"][indices]
    deltas = parent_arrays["candidate_scaled_deltas"][indices]
    coordinates = parent_arrays["candidate_departure_coordinates"][indices]
    base_rate = np.asarray(generator["fixed_Q_rate"], dtype=float)
    if (
        states.shape != (manifest.PLANNED_RATE_EVALUATIONS, 112, 5)
        or deltas.shape != (manifest.PLANNED_RATE_EVALUATIONS, 560)
        or coordinates.shape != (manifest.PLANNED_RATE_EVALUATIONS, 28)
        or geometry["departure_coordinate_basis"].shape != (560, 28)
        or geometry["stable_memory_coordinate_basis"].shape != (560, 280)
        or generator["complete_fixed_Q_generator"].shape != (560, 560)
        or base_rate.shape != (560,)
        or not np.array_equal(coordinates, frozen["design"]["departure_coordinates"])
        or not np.array_equal(base_rate, geometry["fixed_Q_rate"])
    ):
        raise RuntimeError("targeted exact-rate input dimensions changed")
    return {
        "states": states,
        "deltas": deltas,
        "coordinates": coordinates,
        "parent_candidate_indices": indices,
        "direction_indices": np.asarray(frozen["design"]["direction_indices"], dtype=int),
        "signs": np.asarray(frozen["design"]["signs"], dtype=int),
        "candidate_labels": tuple(frozen["design_json"]["candidate_labels"]),
        "departure_basis": np.asarray(geometry["departure_coordinate_basis"], dtype=float),
        "memory_basis": np.asarray(geometry["stable_memory_coordinate_basis"], dtype=float),
        "generator": np.asarray(generator["complete_fixed_Q_generator"], dtype=float),
        "base_rate": base_rate,
        "coefficients": coefficients,
    }


def _progress_array_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "total_rates_per_second": (560,),
        "free_rates_per_second": (560,),
        "physical_reaction_actions_per_second": (560,),
        "multiplier_coordinates_per_second": (3,),
        "online_470_coordinate_rates_per_second": (470,),
        "exact_departure_rates_per_second": (28,),
        "departure_rate_increments_per_second": (28,),
        "departure_linear_references_per_second": (28,),
        "predicted_departure_rates_per_second": (28,),
        "predicted_nonlinear_departure_rates_per_second": (28,),
    }


def _progress_identity() -> dict:
    return {
        "work_package": WORK_PACKAGE,
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_hashes": _checksums(manifest.CANONICAL_DIRECTORY),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "parent_arrays_sha256": _sha(manifest.PARENT_ARRAYS),
        "old_coefficients_sha256": _sha(manifest.OLD_COEFFICIENTS),
    }


def _empty_progress() -> dict:
    progress = {
        "identity": _progress_identity(),
        "evaluations": [],
        "failures": [],
    }
    for name, shape in _progress_array_shapes().items():
        progress[name] = np.empty((0,) + shape, dtype=float)
    return progress


def _save_progress(progress: dict) -> None:
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
        raise RuntimeError("targeted rate scratch checkpoint is incomplete")
    recorded = _read(PROGRESS_JSON)
    if recorded["identity"] != _progress_identity():
        raise RuntimeError("targeted rate scratch identity changed")
    progress = {
        "identity": recorded["identity"],
        "evaluations": recorded["evaluations"],
        "failures": recorded["failures"],
        **_load_npz(PROGRESS_NPZ),
    }
    count = len(progress["evaluations"])
    if any(progress[name].shape != (count,) + shape for name, shape in _progress_array_shapes().items()):
        raise RuntimeError("targeted rate scratch dimensions changed")
    if [item["local_candidate_index"] for item in progress["evaluations"]] != list(range(count)):
        raise RuntimeError("targeted rate scratch ordering changed")
    return progress


def _evaluate_one(
    inputs: dict, progress: dict, index: int, data: dict, components: dict
) -> None:
    state = inputs["states"][index]
    item, arrays = rate_engine.manifest.prior_screen._continuous_rate(data, state)
    coordinate_jacobian, coordinate_metrics = (
        manifest.parent.prior_geometry.chart_tools._coordinate_jacobian(
            state, components
        )
    )
    total_rate = np.asarray(arrays["total_rate"], dtype=float)
    increment = total_rate - inputs["base_rate"]
    linear = inputs["generator"] @ inputs["deltas"][index]
    departure_total = inputs["departure_basis"].T @ total_rate
    departure_increment = inputs["departure_basis"].T @ increment
    departure_linear = inputs["departure_basis"].T @ linear
    nonlinear_truth = departure_increment - departure_linear
    predicted_nonlinear = manifest.old_rate._predict_rate(
        inputs["coordinates"][index], inputs["coefficients"]
    )
    predicted_total = (
        inputs["departure_basis"].T @ inputs["base_rate"]
        + departure_linear
        + predicted_nonlinear
    )
    coordinate = inputs["coordinates"][index]
    radius = float(np.linalg.norm(coordinate))
    exact_norm = float(np.linalg.norm(departure_total))
    predicted_norm = float(np.linalg.norm(predicted_total))
    radial_speed = float(coordinate @ departure_total / radius)
    predicted_radial_speed = float(coordinate @ predicted_total / radius)
    item.update(
        {
            "local_candidate_index": index,
            "parent_candidate_index": int(inputs["parent_candidate_indices"][index]),
            "candidate_label": inputs["candidate_labels"][index],
            "direction_index": int(inputs["direction_indices"][index]),
            "sign": int(inputs["signs"][index]),
            "departure_coordinate_norm": radius,
            "exact_departure_rate_norm_per_second": exact_norm,
            "exact_radial_speed_per_second": radial_speed,
            "exact_radial_growth_per_second": float(
                coordinate @ departure_total / radius**2
            ),
            "exact_radial_direction_cosine": float(
                radial_speed / max(exact_norm, np.finfo(float).tiny)
            ),
            "predicted_radial_speed_per_second": predicted_radial_speed,
            "predicted_radial_direction_cosine": float(
                predicted_radial_speed
                / max(predicted_norm, np.finfo(float).tiny)
            ),
            "full_departure_rate_relative_error": _relative_error(
                predicted_total, departure_total
            ),
            "departure_rate_increment_relative_error": _relative_error(
                departure_linear + predicted_nonlinear, departure_increment
            ),
            "nonlinear_departure_rate_relative_error": _relative_error(
                predicted_nonlinear, nonlinear_truth
            ),
            "coordinate_Jacobian_rank": coordinate_metrics["rank"],
            "coordinate_Jacobian_condition_number": coordinate_metrics[
                "condition_number"
            ],
        }
    )
    online_rate = np.concatenate(
        (
            coordinate_jacobian @ total_rate,
            inputs["memory_basis"].T @ total_rate,
            departure_total,
        )
    )
    progress["evaluations"].append(item)
    values = {
        "total_rates_per_second": total_rate,
        "free_rates_per_second": arrays["free_rate"],
        "physical_reaction_actions_per_second": arrays["reaction_action"],
        "multiplier_coordinates_per_second": arrays["multiplier"],
        "online_470_coordinate_rates_per_second": online_rate,
        "exact_departure_rates_per_second": departure_total,
        "departure_rate_increments_per_second": departure_increment,
        "departure_linear_references_per_second": departure_linear,
        "predicted_departure_rates_per_second": predicted_total,
        "predicted_nonlinear_departure_rates_per_second": predicted_nonlinear,
    }
    for name, value in values.items():
        progress[name] = _append(progress[name], value)


def _execute(inputs: dict) -> tuple[dict, dict[str, np.ndarray]]:
    progress = _load_or_create_progress()
    resumed = len(progress["evaluations"])
    data = rate_engine.manifest.prior_screen.manifest.parent.manifest.failed_screen._anchor_data(
        "primary"
    )
    components = manifest.parent.prior_geometry._prepare_components()
    began = time.perf_counter()
    for index in range(resumed, manifest.PLANNED_RATE_EVALUATIONS):
        try:
            _evaluate_one(inputs, progress, index, data, components)
            status = "accepted"
        except Exception as error:  # fail closed on the first exact-rate failure
            progress["failures"].append(
                {
                    "local_candidate_index": index,
                    "parent_candidate_index": int(inputs["parent_candidate_indices"][index]),
                    "candidate_label": inputs["candidate_labels"][index],
                    "reason": type(error).__name__,
                    "message": str(error),
                }
            )
            status = "failed"
        _save_progress(progress)
        print(
            json.dumps(
                {
                    "exact_rate_evaluation": index + 1,
                    "total": manifest.PLANNED_RATE_EVALUATIONS,
                    "candidate": inputs["candidate_labels"][index],
                    "status": status,
                    "elapsed_this_process_seconds": time.perf_counter() - began,
                }
            ),
            flush=True,
        )
        if progress["failures"]:
            break

    evaluations = progress["evaluations"]

    def maximum(name: str, default=math.inf) -> float:
        values = [item[name] for item in evaluations]
        return float(max(values)) if values else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        values = [item[name] for item in evaluations]
        return float(min(values)) if values else float(default)

    metrics = {
        "planned_nonbase_rate_evaluations": manifest.PLANNED_RATE_EVALUATIONS,
        "completed_nonbase_rate_evaluations": len(evaluations),
        "failed_rate_evaluations": len(progress["failures"]),
        "failures": progress["failures"],
        "minimum_reconstruction_factor": minimum("minimum_reconstruction_factor", math.inf),
        "maximum_reconstruction_factor": maximum("maximum_reconstruction_factor"),
        "maximum_raw_Schur_condition_number": maximum("raw_Schur_condition_number"),
        "maximum_reaction_identity_defect": maximum("reaction_identity_defect"),
        "maximum_rate_tangency_relative_defect": maximum("rate_tangency_relative_defect"),
        "maximum_coordinate_Jacobian_condition_number": maximum("coordinate_Jacobian_condition_number"),
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum("minimum_scattering_optical_depth"),
        "maximum_incoming_excision_characteristics": maximum("incoming_excision_characteristics"),
        "maximum_full_departure_rate_relative_error": maximum("full_departure_rate_relative_error"),
        "median_full_departure_rate_relative_error": float(
            np.median([item["full_departure_rate_relative_error"] for item in evaluations])
        ) if evaluations else math.inf,
        "maximum_departure_rate_increment_relative_error": maximum("departure_rate_increment_relative_error"),
        "maximum_nonlinear_departure_rate_relative_error": maximum("nonlinear_departure_rate_relative_error"),
        "outward_candidate_count": int(sum(item["exact_radial_speed_per_second"] > 0.0 for item in evaluations)),
        "inward_candidate_count": int(sum(item["exact_radial_speed_per_second"] < 0.0 for item in evaluations)),
        "radial_sign_disagreement_count": int(
            sum(
                np.sign(item["exact_radial_speed_per_second"])
                != np.sign(item["predicted_radial_speed_per_second"])
                for item in evaluations
            )
        ),
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "resumed_evaluation_count": resumed,
        "wall_seconds_this_process": time.perf_counter() - began,
        "evaluations": evaluations,
    }
    arrays = {
        "candidate_primitive_states": inputs["states"][: len(evaluations)],
        "candidate_scaled_deltas": inputs["deltas"][: len(evaluations)],
        "candidate_departure_coordinates": inputs["coordinates"][: len(evaluations)],
        "parent_candidate_indices": inputs["parent_candidate_indices"][: len(evaluations)],
        "direction_indices": inputs["direction_indices"][: len(evaluations)],
        "signs": inputs["signs"][: len(evaluations)],
        "base_fixed_Q_rate_per_second": inputs["base_rate"],
        **{name: progress[name] for name in _progress_array_shapes()},
    }
    return metrics, arrays


def _truth_checks(metrics: dict, gates: dict) -> dict:
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
        "coordinate_condition": metrics["maximum_coordinate_Jacobian_condition_number"]
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
    }


def _classify(
    *, truth_passed: bool, forward_cosine: float, old_field_supported: bool, contract: dict
) -> tuple[str, str | None, str]:
    if not truth_passed:
        return FAIL_CLASSIFICATION, None, "not_evaluated"
    threshold = contract["forward_boundary_decision"][
        "radial_direction_cosine_threshold"
    ]
    if forward_cosine >= threshold:
        authorized = (
            "definitions_only_authentic_trajectory_recentered_chart_manifest"
            if old_field_supported
            else "definitions_only_local_rate_extension_and_recentered_chart_manifest"
        )
        return OUTWARD_CLASSIFICATION, authorized, "outward"
    if forward_cosine <= -threshold:
        authorized = (
            "definitions_only_bounded_transient_saturation_validation_manifest"
            if old_field_supported
            else "definitions_only_local_rate_extension_before_saturation_validation_manifest"
        )
        return INWARD_CLASSIFICATION, authorized, "inward"
    return (
        TANGENTIAL_CLASSIFICATION,
        "definitions_only_local_rate_extension_before_trajectory_manifest",
        "nearly_tangential",
    )


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
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("targeted exact-rate screen already canonicalized")
    inputs = _load_inputs(frozen)
    metrics, arrays = _execute(inputs)
    checks = _truth_checks(
        metrics, frozen["contract"]["binding_exact_rate_gates"]
    )
    truth_passed = all(checks.values())
    old_gate = frozen["contract"]["frozen_old_field_diagnostic"][
        "maximum_full_departure_rate_relative_error"
    ]
    old_field_supported = bool(
        truth_passed
        and metrics["maximum_full_departure_rate_relative_error"] <= old_gate
    )
    forward = {}
    forward_cosine = math.nan
    if len(metrics["evaluations"]) == manifest.PLANNED_RATE_EVALUATIONS:
        forward = metrics["evaluations"][manifest.FORWARD_POSITIVE_LOCAL_INDEX]
        forward_cosine = forward["exact_radial_direction_cosine"]
    classification, authorized_next, forward_behavior = _classify(
        truth_passed=truth_passed,
        forward_cosine=forward_cosine,
        old_field_supported=old_field_supported,
        contract=frozen["contract"],
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": truth_passed,
        "certified_component_bound": manifest.CERTIFIED_COMPONENT_BOUND,
        "completed_exact_rate_evaluations": metrics[
            "completed_nonbase_rate_evaluations"
        ],
        "failed_rate_evaluations": metrics["failed_rate_evaluations"],
        "forward_boundary_behavior": forward_behavior,
        "forward_radial_direction_cosine": forward_cosine,
        "forward_radial_speed_per_second": forward.get(
            "exact_radial_speed_per_second"
        ),
        "old_departure28_field_supported_to_0p015": old_field_supported,
        "maximum_old_field_full_departure_rate_relative_error": metrics[
            "maximum_full_departure_rate_relative_error"
        ],
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "stable_memory_remains_dynamic": True,
        "stationary_fast_graph_used": False,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "rate_metrics.json",
        {
            "truth_checks": checks,
            "truth_passed": truth_passed,
            "old_field_error_gate": old_gate,
            "old_field_supported": old_field_supported,
            "forward_boundary_behavior": forward_behavior,
            **metrics,
        },
    )
    _write_npz(CANONICAL_DIRECTORY / "rate_arrays.npz", arrays)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
            "parent_arrays_sha256": _sha(manifest.PARENT_ARRAYS),
            "old_coefficients_sha256": _sha(manifest.OLD_COEFFICIENTS),
            "online_geometry_sha256": _sha(manifest.ONLINE_GEOMETRY),
            "generator_sha256": _sha(manifest.GENERATOR),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.parent.THIS_RUNNER,
        manifest.parent.THIS_TEST,
        manifest.old_rate.THIS_RUNNER,
        manifest.old_rate.THIS_TEST,
        rate_engine.THIS_RUNNER,
        rate_engine.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if truth_passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "resumed_from_evaluation_count": metrics["resumed_evaluation_count"],
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "exact_rate_engine": rate_engine.manifest.prior_screen.THIS_RUNNER,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in manifest.parent.manifest.parent.vector_field.THREAD_ENVIRONMENT
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
                "# Certified-0.015 departure-rate screen WP10c9d6c7c3b5c4f25cf",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{summary['completed_exact_rate_evaluations']}` of `{manifest.PLANNED_RATE_EVALUATIONS}` exact continuous-rate evaluations with `{summary['failed_rate_evaluations']}` failures.",
                "",
                f"The accepted-forward positive boundary state is `{forward_behavior}`: radial-direction cosine `{forward_cosine:.6e}` and radial speed `{forward.get('exact_radial_speed_per_second', math.nan):.6e}` per second.",
                "",
                f"The frozen departure-28 field support flag at 0.015 is `{old_field_supported}`; its maximum full departure-rate relative error is `{metrics['maximum_full_departure_rate_relative_error']:.6e}` against the prospective `0.15` gate.",
                "",
                f"Authorized next artifact: `{authorized_next}`. The 280D stable memory remains dynamic; no state was propagated and no physical microburst, cycle evolution, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True), flush=True)
    if not truth_passed:
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
