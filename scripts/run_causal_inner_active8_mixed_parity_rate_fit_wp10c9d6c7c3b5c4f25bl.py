#!/usr/bin/env python3
"""Evaluate and fit the active-8 mixed parity closure and decoder."""

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

import run_causal_inner_active8_mixed_geometry_preflight_wp10c9d6c7c3b5c4f25bk as parent  # noqa: E402
import run_causal_inner_active8_mixed_parity_database_manifest_wp10c9d6c7c3b5c4f25bj as manifest  # noqa: E402
import run_causal_inner_expanded_departure_rate_screen_wp10c9d6c7c3b5c4f25be as rate_tools  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bl"
GEOMETRY_COMMIT = "ca0d5daebd30ba3bfde518ce75d3b380cd4f56b6"
GEOMETRY_PARENT = "d4a7453aed3cfe29675fb340842028a9283a8aea"
GEOMETRY_TREE = "7b722df4cff21d04693c525533d905d0f83d6950"

PASS_CLASSIFICATION = (
    "active8_mixed_nonlinear_closure_and_decoder_locally_validated"
)
MODEL_FAIL_CLASSIFICATION = (
    "active8_mixed_model_validation_failed_adaptive_database_extension_required"
)
TRUTH_FAIL_CLASSIFICATION = (
    "active8_mixed_truth_database_failed_closure_identification_blocked"
)

ARTIFACT = (
    "causal_inner_active8_mixed_parity_rate_fit_"
    "wp10c9d6c7c3b5c4f25bl"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_mixed_parity_rate_fit_"
    "wp10c9d6c7c3b5c4f25bl.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_mixed_parity_rate_fit_"
    "wp10c9d6c7c3b5c4f25bl.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_MIXED_PARITY_RATE_"
    "FIT_WP10C9D6C7C3B5C4F25BL_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
DATABASE_PATH = parent.CANONICAL_DIRECTORY / "mixed_geometry_database.npz"
GENERATOR_PATH = rate_tools.manifest.GENERATOR_PATH


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


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


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


def _validate_geometry(*, require_clean: bool) -> dict:
    if GEOMETRY_COMMIT.startswith("TO_BE_FILLED"):
        raise RuntimeError("geometry result lineage has not been frozen")
    if _git("rev-parse", GEOMETRY_COMMIT) != GEOMETRY_COMMIT:
        raise RuntimeError("mixed-geometry result commit changed")
    if _git("rev-parse", f"{GEOMETRY_COMMIT}^") != GEOMETRY_PARENT:
        raise RuntimeError("mixed-geometry result lineage changed")
    if _git("rev-parse", f"{GEOMETRY_COMMIT}^{{tree}}") != GEOMETRY_TREE:
        raise RuntimeError("mixed-geometry result tree changed")
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
        raise RuntimeError("mixed-geometry rate-fit authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"geometry source changed: {relative}")
    _checksums(manifest.ARTIFACT_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("mixed parity rate fit requires a clean tracked tree")
    for name, expected in parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(expected))
        / max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    )


def _append(array: np.ndarray, value, *, dtype=float) -> np.ndarray:
    item = np.asarray(value, dtype=dtype)
    return np.concatenate((array, item.reshape((1,) + item.shape)), axis=0)


def _load_inputs() -> dict:
    database = _load_npz(DATABASE_PATH)
    geometry = _load_npz(manifest.GEOMETRY_PATH)
    generator_data = _load_npz(GENERATOR_PATH)
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
        or generator_data["complete_fixed_Q_generator"].shape != (560, 560)
        or geometry["online_coordinate_restriction"].shape != (470, 560)
        or geometry["hidden_stable_remainder_basis"].shape != (560, 90)
    ):
        raise RuntimeError("mixed parity database dimensions changed")
    if [item["candidate_index"] for item in candidates] != list(
        range(manifest.PLANNED_CANDIDATES)
    ):
        raise RuntimeError("mixed parity candidate ordering changed")
    return {
        "database": database,
        "geometry": geometry,
        "generator": np.asarray(
            generator_data["complete_fixed_Q_generator"], dtype=float
        ),
        "base_rate": np.asarray(generator_data["fixed_Q_rate"], dtype=float),
        "states": states,
        "deltas": deltas,
        "coordinates": coordinates,
        "candidates": candidates,
    }


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
        raise RuntimeError("mixed parity rate scratch checkpoint is incomplete")
    recorded = _read(json_path)
    if recorded["identity"] != identity:
        raise RuntimeError("mixed parity rate scratch identity changed")
    progress = {
        "identity": identity,
        "evaluations": recorded["evaluations"],
        "failures": recorded["failures"],
        **_load_npz(npz_path),
    }
    count = len(progress["evaluations"])
    if any(progress[name].shape[0] != count for name in _progress_array_names()):
        raise RuntimeError("mixed parity rate scratch dimensions changed")
    return progress


def _truth_evaluations(inputs: dict) -> tuple[dict, dict[str, np.ndarray]]:
    data = rate_tools.manifest.prior_screen.manifest.parent.manifest.failed_screen._anchor_data(
        "primary"
    )
    components = parent.high_chart._prepare_components()
    progress = _load_or_create_progress()
    resumed_candidate_count = len(progress["evaluations"])
    began = time.perf_counter()
    for index in range(resumed_candidate_count, manifest.PLANNED_CANDIDATES):
        state = inputs["states"][index]
        candidate = inputs["candidates"][index]
        try:
            item, arrays = rate_tools.manifest.prior_screen._continuous_rate(data, state)
            coordinate_jacobian, coordinate_metrics = (
                parent.chart_tools._coordinate_jacobian(state, components)
            )
            linear = inputs["generator"] @ inputs["deltas"][index]
            increment = arrays["total_rate"] - inputs["base_rate"]
            departure_increment = (
                inputs["geometry"]["departure_coordinate_basis"].T @ increment
            )
            departure_linear = (
                inputs["geometry"]["departure_coordinate_basis"].T @ linear
            )
            online_rate = np.concatenate(
                (
                    coordinate_jacobian @ arrays["total_rate"],
                    inputs["geometry"]["stable_memory_coordinate_basis"].T
                    @ arrays["total_rate"],
                    inputs["geometry"]["departure_coordinate_basis"].T
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
                    "state_rate_linear_relative_defect": _relative_error(
                        increment, linear
                    ),
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
                (
                    "physical_reaction_actions_per_second",
                    arrays["reaction_action"],
                ),
                ("multiplier_coordinates_per_second", arrays["multiplier"]),
                ("online_470_coordinate_rates_per_second", online_rate),
                ("departure_rate_increments_per_second", departure_increment),
                ("linear_rate_references_per_second", linear),
                ("departure_linear_references_per_second", departure_linear),
            ):
                progress[name] = _append(progress[name], value)
            status = "accepted"
        except Exception as error:  # fail closed on the first truth failure
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
                    "candidate": index + 1,
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
        "resumed_candidate_count": resumed_candidate_count,
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
        "total_truth_wall_seconds_this_process": time.perf_counter() - began,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "evaluations": evaluations,
    }
    arrays = {name: progress[name] for name in _progress_array_names()}
    return metrics, arrays


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
        "incoming_excision": metrics[
            "maximum_incoming_excision_characteristics"
        ]
        == gates["maximum_incoming_excision_characteristics_equal"],
        "generator_budget": metrics["new_complete_generator_assemblies"]
        == gates["new_complete_generator_assemblies_equal"],
        "root_budget": metrics["new_nonlinear_roots"]
        == gates["new_nonlinear_roots_equal"],
        "propagation_budget": metrics["propagated_states"]
        == gates["propagated_states_equal"],
    }


def _fit_coefficients(
    training_directions: np.ndarray,
    rate_quadratic_targets: np.ndarray,
    rate_cubic_targets: np.ndarray,
    decoder_cubic_targets: np.ndarray,
    decoder_quartic_targets: np.ndarray,
) -> dict[str, np.ndarray]:
    quadratic_features = manifest._quadratic_features(training_directions)
    cubic_kernel = (training_directions @ training_directions.T) ** 3
    quartic_kernel = (training_directions @ training_directions.T) ** 4
    return {
        "rate_quadratic_coefficients": np.linalg.lstsq(
            quadratic_features, rate_quadratic_targets, rcond=None
        )[0],
        "rate_cubic_kernel_coefficients": np.linalg.solve(
            cubic_kernel, rate_cubic_targets
        ),
        "decoder_cubic_kernel_coefficients": np.linalg.solve(
            cubic_kernel, decoder_cubic_targets
        ),
        "decoder_quartic_kernel_coefficients": np.linalg.solve(
            quartic_kernel, decoder_quartic_targets
        ),
    }


def _predict(
    active_coordinates: np.ndarray,
    training_directions: np.ndarray,
    coefficients: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(active_coordinates, dtype=float).reshape(1, -1)
    quadratic = manifest._quadratic_features(active)[0]
    cubic_kernel = (active @ training_directions.T)[0] ** 3
    quartic_kernel = (active @ training_directions.T)[0] ** 4
    rate = (
        quadratic @ coefficients["rate_quadratic_coefficients"]
        + cubic_kernel @ coefficients["rate_cubic_kernel_coefficients"]
    )
    hidden = (
        cubic_kernel @ coefficients["decoder_cubic_kernel_coefficients"]
        + quartic_kernel @ coefficients["decoder_quartic_kernel_coefficients"]
    )
    return rate, hidden


def _fit_and_validate(
    inputs: dict, truth_arrays: dict[str, np.ndarray]
) -> tuple[dict, dict[str, np.ndarray]]:
    energy = np.asarray(inputs["database"]["energy_directions"], dtype=float)
    hidden_basis = np.asarray(
        inputs["geometry"]["hidden_stable_remainder_basis"], dtype=float
    )
    restriction = np.asarray(
        inputs["geometry"]["online_coordinate_restriction"], dtype=float
    )
    lifting = np.asarray(inputs["geometry"]["online_coordinate_lifting"], dtype=float)
    nonlinear = (
        truth_arrays["departure_rate_increments_per_second"]
        - truth_arrays["departure_linear_references_per_second"]
    )
    hidden_coordinates = inputs["deltas"] @ hidden_basis

    pair_directions = []
    pair_radii = []
    pair_split_codes = []
    rate_quadratic_targets = []
    rate_cubic_targets = []
    decoder_cubic_targets = []
    decoder_quartic_targets = []
    split_codes = {"training": 0, "tuning_high": 1, "holdout": 2, "tuning_low": 3}
    for pair_index in range(manifest.PLANNED_CANDIDATES // 2):
        negative = 2 * pair_index
        positive = negative + 1
        left = inputs["candidates"][negative]
        right = inputs["candidates"][positive]
        if (
            left["pair_index"] != pair_index
            or right["pair_index"] != pair_index
            or left["sign"] != -1
            or right["sign"] != 1
            or left["split"] != right["split"]
        ):
            raise RuntimeError("mixed parity signed-pair ordering changed")
        active_negative = energy.T @ inputs["coordinates"][negative]
        active_positive = energy.T @ inputs["coordinates"][positive]
        active_odd = 0.5 * (active_positive - active_negative)
        radius = float(np.linalg.norm(active_odd))
        if radius <= np.finfo(float).tiny:
            raise RuntimeError("mixed parity active radius vanished")
        pair_directions.append(active_odd / radius)
        pair_radii.append(radius)
        pair_split_codes.append(split_codes[left["split"]])
        rate_quadratic_targets.append(
            0.5 * (nonlinear[positive] + nonlinear[negative]) / radius**2
        )
        rate_cubic_targets.append(
            0.5 * (nonlinear[positive] - nonlinear[negative]) / radius**3
        )
        decoder_cubic_targets.append(
            0.5
            * (hidden_coordinates[positive] - hidden_coordinates[negative])
            / radius**3
        )
        decoder_quartic_targets.append(
            0.5
            * (hidden_coordinates[positive] + hidden_coordinates[negative])
            / radius**4
        )
    pair_directions = np.asarray(pair_directions, dtype=float)
    pair_radii = np.asarray(pair_radii, dtype=float)
    pair_split_codes = np.asarray(pair_split_codes, dtype=np.int64)
    rate_quadratic_targets = np.asarray(rate_quadratic_targets, dtype=float)
    rate_cubic_targets = np.asarray(rate_cubic_targets, dtype=float)
    decoder_cubic_targets = np.asarray(decoder_cubic_targets, dtype=float)
    decoder_quartic_targets = np.asarray(decoder_quartic_targets, dtype=float)
    training = pair_split_codes == 0
    training_directions = pair_directions[training]
    coefficients = _fit_coefficients(
        training_directions,
        rate_quadratic_targets[training],
        rate_cubic_targets[training],
        decoder_cubic_targets[training],
        decoder_quartic_targets[training],
    )

    components = parent.high_chart._prepare_components()
    validation = []
    predicted_nonlinear_rates = np.full(
        (manifest.PLANNED_CANDIDATES, 28), np.nan, dtype=float
    )
    predicted_hidden_coordinates = np.full(
        (manifest.PLANNED_CANDIDATES, 90), np.nan, dtype=float
    )
    predicted_scaled_deltas = np.full(
        (manifest.PLANNED_CANDIDATES, 560), np.nan, dtype=float
    )
    for index, candidate in enumerate(inputs["candidates"]):
        if candidate["split"] == "training":
            continue
        active = energy.T @ inputs["coordinates"][index]
        predicted_nonlinear, predicted_hidden = _predict(
            active, training_directions, coefficients
        )
        predicted_increment = (
            truth_arrays["departure_linear_references_per_second"][index]
            + predicted_nonlinear
        )
        online = restriction @ inputs["deltas"][index]
        predicted_delta = lifting @ online + hidden_basis @ predicted_hidden
        state = components["state"] + (
            components["columns"].ravel() * predicted_delta
        ).reshape(components["state"].shape)
        coordinate_value, coordinate_factors = parent.chart_tools._coordinate_value_with_factors(
            state, components
        )
        state_audit = parent.chart_tools._state_audit(components["context"], state)
        item = {
            "candidate_index": index,
            "pair_index": candidate["pair_index"],
            "split": candidate["split"],
            "amplitude_label": candidate["amplitude_label"],
            "sign": candidate["sign"],
            "departure_rate_relative_error": _relative_error(
                predicted_nonlinear,
                nonlinear[index],
            ),
            "full_departure_rate_increment_relative_error": _relative_error(
                predicted_increment,
                truth_arrays["departure_rate_increments_per_second"][index],
            ),
            "hidden_decoder_relative_error": _relative_error(
                predicted_hidden, hidden_coordinates[index]
            ),
            "full_scaled_state_decoder_relative_error": _relative_error(
                predicted_delta, inputs["deltas"][index]
            ),
            "reconstructed_C_phys_residual_infinity": float(
                np.max(np.abs(coordinate_value - components["coordinate_target"]))
            ),
            "minimum_reconstructed_state_reconstruction_factor": min(
                float(np.min(coordinate_factors)),
                state_audit["minimum_reconstruction_factor"],
            ),
            "maximum_reconstructed_H_over_R": state_audit["maximum_H_over_R"],
            "minimum_reconstructed_scattering_optical_depth": state_audit[
                "minimum_scattering_optical_depth"
            ],
        }
        validation.append(item)
        predicted_nonlinear_rates[index] = predicted_nonlinear
        predicted_hidden_coordinates[index] = predicted_hidden
        predicted_scaled_deltas[index] = predicted_delta

    def subset(split: str) -> list[dict]:
        if split == "tuning":
            return [item for item in validation if item["split"].startswith("tuning")]
        return [item for item in validation if item["split"] == split]

    def aggregate(items: list[dict], field: str, operation) -> float:
        return float(operation([item[field] for item in items]))

    tuning = subset("tuning")
    holdout = subset("holdout")
    combined = tuning + holdout
    metrics = {
        "training_pair_count": int(np.count_nonzero(training)),
        "tuning_candidate_count": len(tuning),
        "holdout_candidate_count": len(holdout),
        "training_quadratic_feature_rank": int(
            np.linalg.matrix_rank(manifest._quadratic_features(training_directions))
        ),
        "training_quadratic_feature_condition_number": float(
            np.linalg.cond(manifest._quadratic_features(training_directions))
        ),
        "training_cubic_kernel_condition_number": float(
            np.linalg.cond((training_directions @ training_directions.T) ** 3)
        ),
        "training_quartic_kernel_condition_number": float(
            np.linalg.cond((training_directions @ training_directions.T) ** 4)
        ),
        "tuning_median_departure_rate_relative_error": aggregate(
            tuning, "departure_rate_relative_error", np.median
        ),
        "tuning_maximum_departure_rate_relative_error": aggregate(
            tuning, "departure_rate_relative_error", np.max
        ),
        "holdout_median_departure_rate_relative_error": aggregate(
            holdout, "departure_rate_relative_error", np.median
        ),
        "holdout_maximum_departure_rate_relative_error": aggregate(
            holdout, "departure_rate_relative_error", np.max
        ),
        "tuning_median_hidden_decoder_relative_error": aggregate(
            tuning, "hidden_decoder_relative_error", np.median
        ),
        "tuning_maximum_hidden_decoder_relative_error": aggregate(
            tuning, "hidden_decoder_relative_error", np.max
        ),
        "holdout_median_hidden_decoder_relative_error": aggregate(
            holdout, "hidden_decoder_relative_error", np.median
        ),
        "holdout_maximum_hidden_decoder_relative_error": aggregate(
            holdout, "hidden_decoder_relative_error", np.max
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
        "maximum_full_departure_rate_increment_relative_error": aggregate(
            combined, "full_departure_rate_increment_relative_error", np.max
        ),
        "validation": validation,
    }
    arrays = {
        "pair_active_directions": pair_directions,
        "pair_active_radii": pair_radii,
        "pair_split_codes": pair_split_codes,
        "rate_quadratic_targets": rate_quadratic_targets,
        "rate_cubic_targets": rate_cubic_targets,
        "decoder_cubic_targets": decoder_cubic_targets,
        "decoder_quartic_targets": decoder_quartic_targets,
        "training_active_directions": training_directions,
        "predicted_nonlinear_departure_rates_per_second": predicted_nonlinear_rates,
        "predicted_hidden_coordinates": predicted_hidden_coordinates,
        "predicted_scaled_deltas": predicted_scaled_deltas,
        **coefficients,
    }
    return metrics, arrays


def _model_gate_checks(metrics: dict, gates: dict) -> dict:
    return {
        name: metrics[name] <= threshold
        for name, threshold in gates.items()
        if name != "minimum_reconstructed_state_reconstruction_factor"
    } | {
        "minimum_reconstructed_state_reconstruction_factor": metrics[
            "minimum_reconstructed_state_reconstruction_factor"
        ]
        >= gates["minimum_reconstructed_state_reconstruction_factor"]
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
        raise RuntimeError("mixed parity rate fit is already canonicalized")
    inputs = _load_inputs()
    truth_metrics, truth_arrays = _truth_evaluations(inputs)
    truth_checks = _truth_gate_checks(
        truth_metrics, manifest._contract()["binding_truth_rate_gates"]
    )
    model_metrics = {}
    model_arrays = {}
    model_checks = {"truth_database_complete": False}
    if all(truth_checks.values()):
        model_metrics, model_arrays = _fit_and_validate(inputs, truth_arrays)
        model_checks = _model_gate_checks(
            model_metrics,
            manifest._contract()["binding_model_validation_gates"],
        )
    truth_passed = all(truth_checks.values())
    model_passed = truth_passed and all(model_checks.values())
    if not truth_passed:
        classification = TRUTH_FAIL_CLASSIFICATION
    elif not model_passed:
        classification = MODEL_FAIL_CLASSIFICATION
    else:
        classification = PASS_CLASSIFICATION
    authorized_next = (
        "definitions_only_local_470_closure_short_trajectory_manifest"
        if model_passed
        else None
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "metrics.json",
        {
            "truth_checks": truth_checks,
            "model_checks": model_checks,
            "truth": truth_metrics,
            "model": model_metrics,
        },
    )
    _write_npz(
        CANONICAL_DIRECTORY / "mixed_parity_closure.npz",
        {
            "candidate_primitive_states": inputs["states"],
            "candidate_scaled_deltas": inputs["deltas"],
            "candidate_departure_coordinates": inputs["coordinates"],
            "base_fixed_Q_rate_per_second": inputs["base_rate"],
            **truth_arrays,
            **model_arrays,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": model_passed,
        "truth_database_passed": truth_passed,
        "local_closure_validated": model_passed,
        "completed_nonbase_rate_evaluations": truth_metrics[
            "completed_nonbase_rate_evaluations"
        ],
        "failed_rate_evaluations": truth_metrics["failed_rate_evaluations"],
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
            "manifest_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
        },
    )
    source_files = (THIS_RUNNER, THIS_TEST, parent.THIS_RUNNER, parent.THIS_TEST)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if model_passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "resumed_from_candidate_count": truth_metrics[
                "resumed_candidate_count"
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
    _write_json(
        CANONICAL_DIRECTORY / "model_contract.json",
        manifest._contract()["closure_models"],
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    model_lines = (
        (
            f"Tuning rate error median/max: `{model_metrics['tuning_median_departure_rate_relative_error']:.6e}` / `{model_metrics['tuning_maximum_departure_rate_relative_error']:.6e}`. Holdout: `{model_metrics['holdout_median_departure_rate_relative_error']:.6e}` / `{model_metrics['holdout_maximum_departure_rate_relative_error']:.6e}`."
        )
        if model_metrics
        else "The truth database failed before model fitting."
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Active-8 mixed parity rate fit WP10c9d6c7c3b5c4f25bl",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{truth_metrics['completed_nonbase_rate_evaluations']}` of `{manifest.PLANNED_CANDIDATES}` planned nonbase fixed-Q truth-rate evaluations; failures: `{truth_metrics['failed_rate_evaluations']}`.",
                "",
                model_lines,
                "",
                f"Authorized next artifact: `{authorized_next}`. No state was propagated and no predictive cycle or reduced slow evolution is authorized.",
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
