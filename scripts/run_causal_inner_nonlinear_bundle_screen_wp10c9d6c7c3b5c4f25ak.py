#!/usr/bin/env python3
"""Select a nonlinear treatment for the exact positive-growth bundle."""

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

import run_causal_inner_nonlinear_bundle_database_manifest_wp10c9d6c7c3b5c4f25aj as manifest  # noqa: E402
import run_causal_inner_pathwise_closure_descriptor_pilot_wp10c9d6c7c3b5c4f25c as descriptor_tools  # noqa: E402
import run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1 as history_tools  # noqa: E402
from run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a import (  # noqa: E402
    _state_audit,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_exterior_q3,
    causal_five_field_fixed_q_reaction,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_dae import (  # noqa: E402
    evaluate_causal_five_field_monolithic_backward_euler,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ak"
MANIFEST_COMMIT = "97e1c75cb32329e9272a62432a1da6e10e094204"
MANIFEST_PARENT = "8d48e1d19295024e0a81a428dc4b5e379c43cc19"
MANIFEST_TREE = "afbea5662b95124ec675eae9562d3d12a486ed3a"

LOCAL_CLASSIFICATION = (
    "finite_amplitude_local_saturation_supported_"
    "energy_bounded_normal_form_manifest_authorized"
)
HYBRID_CLASSIFICATION = (
    "finite_amplitude_local_saturation_not_supported_"
    "conservative_hybrid_branch_event_architecture_selected"
)
FAIL_CLASSIFICATION = (
    "nonlinear_fixed_Q_bundle_evaluator_failed_"
    "architecture_selection_blocked"
)

ARTIFACT = "causal_inner_nonlinear_bundle_screen_wp10c9d6c7c3b5c4f25ak"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_bundle_screen_"
    "wp10c9d6c7c3b5c4f25ak.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_bundle_screen_"
    "wp10c9d6c7c3b5c4f25ak.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_BUNDLE_SCREEN_"
    "WP10C9D6C7C3B5C4F25AK_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CROSS_ANCHOR_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_common_resolved_subspace_cross_anchor_"
    "preflight_wp10c9d6c7c3b5c4f25o"
)
PRIMARY_GENERATOR_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_pathwise_closure_descriptor_pilot_"
    "wp10c9d6c7c3b5c4f25c"
)


class CoordinateRetractionFailure(RuntimeError):
    """Fail closed when a rate reaction is unusable as a state chart."""

    def __init__(self, diagnostics: dict):
        super().__init__("reaction-lift coordinate retraction left its trust region")
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


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("nonlinear-bundle manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("nonlinear-bundle manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("nonlinear-bundle manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_nonbase_rate_evaluations"] != 96
        or contract["claim_boundary"]["branch_existence_assumed"]
    ):
        raise RuntimeError("nonlinear-bundle execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(manifest.PARENT_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent input changed: {name}")
    for name, expected in contract["fiber_decisive_hashes"].items():
        if _sha(manifest.FIBER_DIRECTORY / name) != expected:
            raise RuntimeError(f"fiber input changed: {name}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("nonlinear-bundle screen requires a clean tracked tree")
    for name, expected in manifest.parent.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(float).tiny)
    )


def _energy_directions(operator: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    symmetric = 0.5 * (np.asarray(operator) + np.asarray(operator).T)
    values, vectors = np.linalg.eigh(symmetric)
    selected = np.asarray(vectors[:, -int(count) :][:, ::-1], dtype=float)
    selected_values = np.asarray(values[-int(count) :][::-1], dtype=float)
    for column in range(selected.shape[1]):
        pivot = int(np.argmax(np.abs(selected[:, column])))
        if selected[pivot, column] < 0.0:
            selected[:, column] *= -1.0
    return selected, selected_values


def _radial_metrics(
    linear_growth: np.ndarray,
    coordinate_radii: np.ndarray,
    projected_rates: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    rates = np.asarray(projected_rates, dtype=float)
    radii = np.asarray(coordinate_radii, dtype=float)
    count, amplitudes, signs, dimension = rates.shape
    if signs != 2 or linear_growth.shape != (count,) or radii.shape != (count, amplitudes):
        raise ValueError("radial-metric shapes are invalid")
    directions = np.zeros((count, dimension), dtype=float)
    central_growth = np.empty((count, amplitudes), dtype=float)
    for index in range(count):
        delta = rates[index, 0, 1] - rates[index, 0, 0]
        norm = float(np.linalg.norm(delta))
        if norm <= np.finfo(float).tiny:
            raise ValueError("screen direction has no odd response")
        # Direction is supplied separately by the caller in production; this
        # helper's synthetic-test convention aligns it with the smallest pair.
        directions[index] = delta / norm
        for amplitude in range(amplitudes):
            odd = 0.5 * (rates[index, amplitude, 1] - rates[index, amplitude, 0])
            central_growth[index, amplitude] = float(
                directions[index] @ odd / radii[index, amplitude]
            )
    cubic = np.empty(count, dtype=float)
    for index in range(count):
        cubic[index] = float(
            np.polyfit(radii[index] ** 2, central_growth[index], 1)[0]
        )
    metrics = {
        "smallest_amplitude_linear_growth_relative_defect": float(
            np.max(
                np.abs(central_growth[:, 0] - linear_growth)
                / np.maximum(np.abs(linear_growth), np.finfo(float).tiny)
            )
        ),
        "nonpositive_largest_amplitude_growth_count": int(
            np.count_nonzero(central_growth[:, -1] <= 0.0)
        ),
        "negative_fitted_cubic_count": int(np.count_nonzero(cubic < 0.0)),
    }
    return metrics, {
        "central_radial_growth_per_second": central_growth,
        "fitted_cubic_growth_coefficients": cubic,
    }


def _anchor_data(anchor: str) -> dict:
    if anchor == "primary":
        data = descriptor_tools._seed_data()
        return {
            "layout": data["layout"],
            "context": data["context"],
            "state": np.asarray(data["state"], dtype=float),
            "columns": np.asarray(data["columns"], dtype=float),
            "rows": np.asarray(data["rows"], dtype=float),
        }
    if anchor != "heldout":
        raise ValueError("unknown nonlinear-screen anchor")
    layout, configuration, trajectory = history_tools.c4f13._layout_data("middle")
    matches = np.flatnonzero(np.asarray(trajectory["times"], dtype=float) == 1.6e-2)
    if matches.size != 1:
        raise RuntimeError("unique held-out state is unavailable")
    state = np.asarray(trajectory["states"][int(matches[0])], dtype=float)
    with np.load(CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz") as source:
        if not np.array_equal(state, np.asarray(source["primitive_state"])):
            raise RuntimeError("held-out generator state changed")
    return {
        "layout": layout,
        "context": configuration["context"],
        "state": state,
        "columns": np.asarray(configuration["columns"], dtype=float).reshape(state.shape),
        "rows": np.asarray(configuration["rows"], dtype=float).reshape(state.shape),
    }


def _continuous_fixed_q_rate(data: dict, state: np.ndarray) -> tuple[np.ndarray, object, object, dict, dict]:
    timings: dict[str, float] = {}
    began = time.perf_counter()
    reaction = causal_five_field_fixed_q_reaction(
        data["context"],
        state,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        maximum_schur_condition_number=1.0e8,
        timing_accumulator=timings,
    )
    evaluation = evaluate_causal_five_field_monolithic_backward_euler(
        state, state, 1.0, data["context"], path_quadrature_order=6
    )
    stationary = np.asarray(evaluation.residual_rows, dtype=float).ravel() / data[
        "rows"
    ].ravel()
    free = np.linalg.solve(reaction.descriptor_scaled_matrix, -stationary)
    multiplier = -reaction.q3_scaled_derivative @ free
    rate = free + reaction.reaction_lift @ multiplier
    physical = _state_audit(data["context"], state)
    timings["total_continuous_rate_wall_seconds"] = time.perf_counter() - began
    return rate, reaction, evaluation, physical, timings


def _retract_to_base_q(
    data: dict,
    base_reaction,
    scaled_trial: np.ndarray,
    *,
    maximum_iterations: int = 12,
    maximum_component: float | None = None,
) -> tuple[np.ndarray, dict]:
    delta = np.asarray(scaled_trial, dtype=float).ravel().copy()
    q0 = np.asarray(base_reaction.q3_value, dtype=float)
    qscale = np.asarray(base_reaction.q3_derivative_norms, dtype=float)
    face = 36 * int(data["layout"].refinement_ratio)
    errors = []
    factors = None
    component_bound = None if maximum_component is None else float(maximum_component)
    for _outer in range(6):
        for _ in range(maximum_iterations):
            state = data["state"] + (data["columns"].ravel() * delta).reshape(
                data["state"].shape
            )
            q3, factors = causal_five_field_exterior_q3(
                data["context"], state, exterior_face_index=face
            )
            error = (np.asarray(q3) - q0) / qscale
            errors.append(float(np.max(np.abs(error))))
            if errors[-1] <= 1.0e-12:
                break
            correction = base_reaction.reaction_lift @ error
            proposed = delta - correction
            if (
                component_bound is not None
                and float(np.max(np.abs(proposed))) > 4.0 * component_bound
            ):
                raise CoordinateRetractionFailure(
                    {
                        "normalized_Q3_error": errors[-1],
                        "maximum_scaled_trial_component": float(
                            np.max(np.abs(delta))
                        ),
                        "maximum_scaled_reaction_correction": float(
                            np.max(np.abs(correction))
                        ),
                        "maximum_scaled_proposed_component": float(
                            np.max(np.abs(proposed))
                        ),
                        "declared_component_bound": component_bound,
                        "reaction_lift_spectral_norm": float(
                            np.linalg.norm(base_reaction.reaction_lift, 2)
                        ),
                        "failure_kind": "rate_reaction_is_not_a_geometric_retraction",
                    }
                )
            delta = proposed
        maximum = float(np.max(np.abs(delta)))
        if component_bound is None or maximum <= component_bound * (1.0 + 1.0e-12):
            break
        delta *= component_bound / maximum
    state = data["state"] + (data["columns"].ravel() * delta).reshape(
        data["state"].shape
    )
    q3, factors = causal_five_field_exterior_q3(
        data["context"], state, exterior_face_index=face
    )
    errors.append(float(np.max(np.abs((np.asarray(q3) - q0) / qscale))))
    return state, {
        "normalized_Q3_retraction_defect": errors[-1],
        "retraction_iterations": len(errors),
        "minimum_Q3_reconstruction_factor": float(np.min(factors)),
        "maximum_Q3_reconstruction_factor": float(np.max(factors)),
        "scaled_delta": delta,
    }


def _screen_anchor(anchor: str, fiber: dict, gates: dict) -> tuple[dict, dict[str, np.ndarray]]:
    data = _anchor_data(anchor)
    state = data["state"]
    right = np.asarray(fiber[f"{anchor}_right_basis"], dtype=float)
    left = np.asarray(fiber[f"{anchor}_left_dual_transpose"], dtype=float)
    operator = np.asarray(fiber[f"{anchor}_unstable_operator"], dtype=float)
    directions, energy_values = _energy_directions(
        operator, manifest.ENERGY_DIRECTIONS
    )
    base_rate, base_reaction, base_evaluation, base_physical, base_timing = (
        _continuous_fixed_q_rate(data, state)
    )
    amplitudes = np.asarray(manifest.MAXIMUM_COMPONENT_AMPLITUDES, dtype=float)
    n_direction = directions.shape[1]
    n_amplitude = amplitudes.size
    radii = np.empty((n_direction, n_amplitude), dtype=float)
    effective_radii = np.empty((n_direction, n_amplitude), dtype=float)
    projected = np.empty((n_direction, n_amplitude, 2, manifest.UNSTABLE_DIMENSION))
    actual_coordinates = np.empty_like(projected)
    q_defects = np.empty((n_direction, n_amplitude, 2), dtype=float)
    component_changes = np.empty_like(q_defects)
    minimum_factors = np.empty_like(q_defects)
    maximum_factors = np.empty_like(q_defects)
    schur_conditions = np.empty_like(q_defects)
    identity_defects = np.empty_like(q_defects)
    tangency_defects = np.empty_like(q_defects)
    h_over_r = np.empty_like(q_defects)
    optical_depth = np.empty_like(q_defects)
    incoming = np.empty_like(q_defects, dtype=int)
    wall_seconds = np.empty_like(q_defects)
    for direction_index in range(n_direction):
        direction = directions[:, direction_index]
        lifted = right @ direction
        maximum = float(np.max(np.abs(lifted)))
        if maximum <= np.finfo(float).tiny:
            raise RuntimeError("unstable direction has zero lifting")
        radii[direction_index] = amplitudes / maximum
        for amplitude_index, radius in enumerate(radii[direction_index]):
            for sign_index, sign in enumerate((-1.0, 1.0)):
                trial = sign * radius * lifted
                candidate, retraction = _retract_to_base_q(
                    data,
                    base_reaction,
                    trial,
                    maximum_component=float(amplitudes[amplitude_index]),
                )
                rate, reaction, evaluation, physical, timing = (
                    _continuous_fixed_q_rate(data, candidate)
                )
                actual_delta = np.asarray(retraction["scaled_delta"])
                projected[direction_index, amplitude_index, sign_index] = (
                    left @ (rate - base_rate)
                )
                actual_coordinates[direction_index, amplitude_index, sign_index] = (
                    left @ actual_delta
                )
                q_defects[direction_index, amplitude_index, sign_index] = (
                    retraction["normalized_Q3_retraction_defect"]
                )
                component_changes[direction_index, amplitude_index, sign_index] = float(
                    np.max(np.abs(actual_delta))
                )
                minimum_factors[direction_index, amplitude_index, sign_index] = min(
                    retraction["minimum_Q3_reconstruction_factor"],
                    reaction.minimum_q3_reconstruction_factor,
                    physical["minimum_reconstruction_factor"],
                )
                maximum_factors[direction_index, amplitude_index, sign_index] = max(
                    retraction["maximum_Q3_reconstruction_factor"],
                    reaction.maximum_q3_reconstruction_factor,
                )
                schur_conditions[direction_index, amplitude_index, sign_index] = (
                    reaction.raw_schur_condition_number
                )
                identity_defects[direction_index, amplitude_index, sign_index] = (
                    reaction.maximum_identity_defect
                )
                tangency_defects[direction_index, amplitude_index, sign_index] = float(
                    np.linalg.norm(reaction.q3_scaled_derivative @ rate)
                    / max(float(np.linalg.norm(rate)), np.finfo(float).tiny)
                )
                h_over_r[direction_index, amplitude_index, sign_index] = physical[
                    "maximum_h_over_r"
                ]
                optical_depth[direction_index, amplitude_index, sign_index] = physical[
                    "minimum_scattering_optical_depth"
                ]
                incoming[direction_index, amplitude_index, sign_index] = (
                    evaluation.incoming_excision_characteristics
                )
                wall_seconds[direction_index, amplitude_index, sign_index] = timing[
                    "total_continuous_rate_wall_seconds"
                ]
                print(
                    json.dumps(
                        {
                            "anchor": anchor,
                            "direction": direction_index,
                            "amplitude": float(amplitudes[amplitude_index]),
                            "sign": int(sign),
                            "completed": int(
                                direction_index * n_amplitude * 2
                                + amplitude_index * 2
                                + sign_index
                                + 1
                            ),
                            "total": int(n_direction * n_amplitude * 2),
                            "wall_seconds": float(wall_seconds[direction_index, amplitude_index, sign_index]),
                        }
                    ),
                    flush=True,
                )
    linear_growth = np.asarray(
        [
            directions[:, index] @ operator @ directions[:, index]
            for index in range(n_direction)
        ]
    )
    central_growth = np.empty((n_direction, n_amplitude), dtype=float)
    cubic = np.empty(n_direction, dtype=float)
    coordinate_symmetry = np.empty((n_direction, n_amplitude), dtype=float)
    for index in range(n_direction):
        direction = directions[:, index]
        for amplitude_index, radius in enumerate(radii[index]):
            effective_radius = float(
                0.5
                * direction
                @ (
                    actual_coordinates[index, amplitude_index, 1]
                    - actual_coordinates[index, amplitude_index, 0]
                )
            )
            if effective_radius <= np.finfo(float).tiny:
                raise RuntimeError("retracted unstable coordinate lost its direction")
            effective_radii[index, amplitude_index] = effective_radius
            odd = 0.5 * (
                projected[index, amplitude_index, 1]
                - projected[index, amplitude_index, 0]
            )
            central_growth[index, amplitude_index] = float(
                direction @ odd / effective_radius
            )
            coordinate_symmetry[index, amplitude_index] = float(
                np.linalg.norm(
                    actual_coordinates[index, amplitude_index, 1]
                    + actual_coordinates[index, amplitude_index, 0]
                )
                / max(
                    float(
                        np.linalg.norm(actual_coordinates[index, amplitude_index, 1])
                        + np.linalg.norm(actual_coordinates[index, amplitude_index, 0])
                    ),
                    np.finfo(float).tiny,
                )
            )
        cubic[index] = float(
            np.polyfit(effective_radii[index] ** 2, central_growth[index], 1)[0]
        )
    smallest_defects = np.abs(central_growth[:, 0] - linear_growth) / np.maximum(
        np.abs(linear_growth), np.finfo(float).tiny
    )
    metrics = {
        "base_rate_norm_per_second": float(np.linalg.norm(base_rate)),
        "base_rate_tangency_relative_defect": float(
            np.linalg.norm(base_reaction.q3_scaled_derivative @ base_rate)
            / max(float(np.linalg.norm(base_rate)), np.finfo(float).tiny)
        ),
        "base_maximum_H_over_R": base_physical["maximum_h_over_r"],
        "base_minimum_scattering_optical_depth": base_physical[
            "minimum_scattering_optical_depth"
        ],
        "base_incoming_excision_characteristics": int(
            base_evaluation.incoming_excision_characteristics
        ),
        "base_rate_wall_seconds": base_timing["total_continuous_rate_wall_seconds"],
        "minimum_linear_energy_growth_per_second": float(np.min(linear_growth)),
        "maximum_linear_energy_growth_per_second": float(np.max(linear_growth)),
        "maximum_smallest_amplitude_linear_growth_relative_defect": float(
            np.max(smallest_defects)
        ),
        "median_smallest_amplitude_linear_growth_relative_defect": float(
            np.median(smallest_defects)
        ),
        "nonpositive_largest_amplitude_growth_count": int(
            np.count_nonzero(central_growth[:, -1] <= 0.0)
        ),
        "negative_fitted_cubic_count": int(np.count_nonzero(cubic < 0.0)),
        "minimum_largest_amplitude_growth_per_second": float(
            np.min(central_growth[:, -1])
        ),
        "maximum_largest_amplitude_growth_per_second": float(
            np.max(central_growth[:, -1])
        ),
        "maximum_normalized_Q3_retraction_defect": float(np.max(q_defects)),
        "maximum_scaled_component_perturbation": float(np.max(component_changes)),
        "minimum_reconstruction_factor": float(np.min(minimum_factors)),
        "maximum_reconstruction_factor": float(np.max(maximum_factors)),
        "maximum_raw_Schur_condition_number": float(np.max(schur_conditions)),
        "maximum_reaction_identity_defect": float(np.max(identity_defects)),
        "maximum_rate_tangency_relative_defect": float(np.max(tangency_defects)),
        "maximum_H_over_R": float(np.max(h_over_r)),
        "minimum_scattering_optical_depth": float(np.min(optical_depth)),
        "maximum_incoming_excision_characteristics": int(np.max(incoming)),
        "maximum_coordinate_odd_symmetry_defect": float(np.max(coordinate_symmetry)),
        "total_nonbase_rate_wall_seconds": float(np.sum(wall_seconds)),
        "median_nonbase_rate_wall_seconds": float(np.median(wall_seconds)),
    }
    metrics["evaluator_passed"] = bool(
        metrics["base_rate_tangency_relative_defect"]
        <= gates["maximum_rate_tangency_relative_defect"]
        and metrics["base_incoming_excision_characteristics"] == 0
        and metrics["maximum_smallest_amplitude_linear_growth_relative_defect"]
        <= gates["maximum_smallest_amplitude_linear_growth_relative_defect"]
        and metrics["maximum_normalized_Q3_retraction_defect"]
        <= gates["maximum_normalized_Q3_retraction_defect"]
        and metrics["maximum_scaled_component_perturbation"]
        <= gates["maximum_scaled_component_perturbation"] * (1.0 + 1.0e-9)
        and metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"]
        and metrics["maximum_reconstruction_factor"]
        <= gates["maximum_reconstruction_factor"]
        and metrics["maximum_raw_Schur_condition_number"]
        <= gates["maximum_raw_Schur_condition_number"]
        and metrics["maximum_reaction_identity_defect"]
        <= gates["maximum_reaction_identity_defect"]
        and metrics["maximum_rate_tangency_relative_defect"]
        <= gates["maximum_rate_tangency_relative_defect"]
        and metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"]
        and metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"]
        and metrics["maximum_incoming_excision_characteristics"] == 0
    )
    selection = manifest._contract()["architecture_selection"][
        "local_saturation_requires"
    ]
    metrics["local_saturation_supported"] = bool(
        metrics["evaluator_passed"]
        and metrics["nonpositive_largest_amplitude_growth_count"]
        >= selection[
            "nonpositive_largest_amplitude_radial_growth_directions_per_anchor_min"
        ]
        and metrics["negative_fitted_cubic_count"]
        >= selection["negative_fitted_cubic_directions_per_anchor_min"]
    )
    arrays = {
        "directions": directions,
        "symmetric_energy_eigenvalues_per_second": energy_values,
        "linear_radial_growth_per_second": linear_growth,
        "maximum_component_amplitudes": amplitudes,
        "coordinate_radii": radii,
        "effective_retracted_coordinate_radii": effective_radii,
        "projected_rate_differences_per_second": projected,
        "actual_unstable_coordinates": actual_coordinates,
        "central_radial_growth_per_second": central_growth,
        "fitted_cubic_growth_coefficients": cubic,
        "normalized_Q3_retraction_defects": q_defects,
        "scaled_component_changes": component_changes,
        "minimum_reconstruction_factors": minimum_factors,
        "maximum_reconstruction_factors": maximum_factors,
        "raw_Schur_condition_numbers": schur_conditions,
        "reaction_identity_defects": identity_defects,
        "rate_tangency_relative_defects": tangency_defects,
        "maximum_H_over_R": h_over_r,
        "minimum_scattering_optical_depth": optical_depth,
        "incoming_excision_characteristics": incoming,
        "continuous_rate_wall_seconds": wall_seconds,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "CERTIFIED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": status,
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
        raise RuntimeError("nonlinear-bundle screen is already canonicalized")
    began = time.perf_counter()
    with np.load(manifest.FIBER_DIRECTORY / "decisive_fibers.npz") as source:
        fiber = {name: np.asarray(source[name]) for name in source.files}
    gates = frozen["contract"]["binding_evaluator_gates"]
    anchor_metrics = {}
    saved_arrays = {}
    failure = None
    for anchor in manifest.ANCHORS:
        try:
            anchor_metrics[anchor], arrays = _screen_anchor(anchor, fiber, gates)
        except CoordinateRetractionFailure as error:
            failure = {"anchor": anchor, **error.diagnostics}
            saved_arrays["failed_retraction_diagnostics"] = np.asarray(
                (
                    error.diagnostics["normalized_Q3_error"],
                    error.diagnostics["maximum_scaled_trial_component"],
                    error.diagnostics["maximum_scaled_reaction_correction"],
                    error.diagnostics["maximum_scaled_proposed_component"],
                    error.diagnostics["declared_component_bound"],
                    error.diagnostics["reaction_lift_spectral_norm"],
                ),
                dtype=float,
            )
            break
        for name, value in arrays.items():
            saved_arrays[f"{anchor}_{name}"] = value
    evaluator_passed = bool(
        failure is None
        and all(
            anchor_metrics[anchor]["evaluator_passed"]
            for anchor in manifest.ANCHORS
        )
    )
    local_supported = bool(
        evaluator_passed
        and all(
            anchor_metrics[anchor]["local_saturation_supported"]
            for anchor in manifest.ANCHORS
        )
    )
    if not evaluator_passed:
        classification = FAIL_CLASSIFICATION
        authorized_next = None
        selected_architecture = None
        passed = False
    elif local_supported:
        classification = LOCAL_CLASSIFICATION
        authorized_next = "definitions_only_energy_bounded_normal_form_identification_manifest"
        selected_architecture = "energy_bounded_nonlinear_normal_form"
        passed = True
    else:
        classification = HYBRID_CLASSIFICATION
        authorized_next = "definitions_only_conservative_hybrid_branch_event_database_manifest"
        selected_architecture = "conservative_hybrid_branch_and_event_map"
        passed = True
    metrics = {
        "anchors": anchor_metrics,
        "evaluator_passed": evaluator_passed,
        "local_saturation_supported": local_supported,
        "selected_architecture": selected_architecture,
        "fail_fast_coordinate_retraction": failure,
        "new_full_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "planned_nonbase_continuous_rate_evaluations": int(
            len(manifest.ANCHORS)
            * manifest.ENERGY_DIRECTIONS
            * 2
            * len(manifest.MAXIMUM_COMPONENT_AMPLITUDES)
        ),
        "completed_nonbase_continuous_rate_evaluations": (
            0
            if failure is not None
            else int(
                len(manifest.ANCHORS)
                * manifest.ENERGY_DIRECTIONS
                * 2
                * len(manifest.MAXIMUM_COMPONENT_AMPLITUDES)
            )
        ),
        "total_wall_seconds": time.perf_counter() - began,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "evaluator_passed": evaluator_passed,
        "local_saturation_supported": local_supported,
        "selected_architecture": selected_architecture,
        "authorized_next": authorized_next,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "nonlinear_screen.npz", **saved_arrays)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
            "fiber_package_hashes": _checksums(manifest.FIBER_DIRECTORY),
            "primary_generator_hashes": _checksums(PRIMARY_GENERATOR_DIRECTORY),
            "cross_anchor_hashes": _checksums(CROSS_ANCHOR_DIRECTORY),
        },
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
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
                manifest.THIS_RUNNER: _sha(ROOT / manifest.THIS_RUNNER),
                manifest.THIS_TEST: _sha(ROOT / manifest.THIS_TEST),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": manifest.parent.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    lines = [
        "# Nonlinear unstable-bundle screen WP10c9d6c7c3b5c4f25ak",
        "",
        "## Classification",
        "",
        f"`{classification}`",
        "",
        f"The exact nonlinear fixed-Q evaluator passed: `{evaluator_passed}`. Local trust-region saturation passed: `{local_supported}`.",
        "",
    ]
    if failure is not None:
        lines.extend(
            (
                "## Fail-fast coordinate diagnosis",
                "",
                f"At `{failure['anchor']}`, a normalized Q3 error of `{failure['normalized_Q3_error']:.6e}` required a reaction-lift state correction with maximum scaled component `{failure['maximum_scaled_reaction_correction']:.6e}`, versus the frozen `{failure['declared_component_bound']:.6e}` trust bound. The reaction-lift spectral norm was `{failure['reaction_lift_spectral_norm']:.6e}`.",
                "",
                "The physical reaction lift is certified for enforcing a rate constraint, but it is not a minimum-norm geometric normal for finite-amplitude state retraction. The screen therefore stopped before admitting any nonlinear sample; this is not evidence against nonlinear saturation or the physical equations.",
                "",
            )
        )
    for anchor in anchor_metrics:
        item = anchor_metrics[anchor]
        lines.extend(
            (
                f"## {anchor}",
                "",
                f"Small-amplitude linear-growth defect: `{item['maximum_smallest_amplitude_linear_growth_relative_defect']:.6e}`. Nonpositive largest-amplitude directions: `{item['nonpositive_largest_amplitude_growth_count']}/8`. Negative fitted cubic directions: `{item['negative_fitted_cubic_count']}/8`.",
                "",
                f"Largest-amplitude radial growth spans `{item['minimum_largest_amplitude_growth_per_second']:.6e}` to `{item['maximum_largest_amplitude_growth_per_second']:.6e} s^-1`. Maximum Q3 retraction defect is `{item['maximum_normalized_Q3_retraction_defect']:.6e}`.",
                "",
            )
        )
    lines.extend(
        (
            "## Decision",
            "",
            f"Selected architecture: `{selected_architecture}`.",
            "",
            f"Authorized next artifact: `{authorized_next}`. No predictive cycle or reduced slow evolution is authorized by this screen.",
            "",
        )
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    _update_catalog(summary)
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
