#!/usr/bin/env python3
"""Freeze a shell-gated degree-4/5 local extension and recentered atlas."""

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

import run_causal_inner_certified_0p015_departure_rate_screen_wp10c9d6c7c3b5c4f25cf as parent  # noqa: E402
import run_causal_inner_departure28_short_vector_field_validation_wp10c9d6c7c3b5c4f25bz as vector_field  # noqa: E402
import run_causal_inner_guarded_departure_amplitude_expansion_manifest_wp10c9d6c7c3b5c4f25cc as direction_source  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cg"
PARENT_COMMIT = "3509fa0c8fba7ac7eb6bc931ef4494e963f458fa"
PARENT_PARENT = "89980b1137075d562441c072bd438e2d3c4f4313"
PARENT_TREE = "de8c317418d9e3dad4bf71c6ed566aa3612f1fc4"
CLASSIFICATION = "shell_gated_degree45_recentered_atlas_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25ch"

EVEN_POWER = 4
ODD_POWER = 5
KERNEL_REGULARIZATION = 1.0e-10
INNER_PRESERVATION_LOAD = 1.0e-2
FULL_EXTENSION_LOAD = 1.3e-2
RECENTER_TRIGGER_LOAD = 1.2e-2
HARD_CHART_LOAD = 1.5e-2
HOLDOUT_COMPONENT_BOUNDS = (1.25e-2, 1.5e-2)
HOLDOUT_MIXING = 0.25
HOLDOUT_DIRECTION_COUNT = 4
PLANNED_GEOMETRY_CANDIDATES = (
    len(HOLDOUT_COMPONENT_BOUNDS) * HOLDOUT_DIRECTION_COUNT
)

ARTIFACT = (
    "causal_inner_shell_gated_recentered_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25cg"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_shell_gated_recentered_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25cg.py"
)
THIS_TEST = (
    "tests/test_causal_inner_shell_gated_recentered_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25cg.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_shell_gated_atlas_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25ch.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_shell_gated_atlas_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25ch.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SHELL_GATED_RECENTERED_"
    "ATLAS_MANIFEST_WP10C9D6C7C3B5C4F25CG_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_ARRAYS = parent.CANONICAL_DIRECTORY / "rate_arrays.npz"
OLD_CLOSURE = parent.manifest.old_rate.CANONICAL_DIRECTORY / "departure28_closure.npz"
DIRECTION_DESIGN = direction_source.CANONICAL_DIRECTORY / "direction_design.npz"

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


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("targeted rate result commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("targeted rate result lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("targeted rate result tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "rate_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["classification"] != parent.OUTWARD_CLASSIFICATION
        or summary["forward_boundary_behavior"] != "outward"
        or summary["forward_radial_direction_cosine"] < 0.99
        or summary["old_departure28_field_supported_to_0p015"]
        or summary["authorized_next"]
        != "definitions_only_local_rate_extension_and_recentered_chart_manifest"
        or summary["completed_exact_rate_evaluations"] != 8
        or summary["new_complete_generator_assemblies"] != 0
        or summary["new_nonlinear_roots"] != 0
        or summary["propagated_states"] != 0
        or not all(metrics["truth_checks"].values())
    ):
        raise RuntimeError("targeted outward-rate authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"targeted rate source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("recentered-atlas manifest requires a clean tracked tree")
    return {"summary": summary, "metrics": metrics, "hashes": hashes}


def _shell_weight(load: float) -> float:
    value = float(load)
    if value <= INNER_PRESERVATION_LOAD:
        return 0.0
    if value >= FULL_EXTENSION_LOAD:
        return 1.0
    t = (value - INNER_PRESERVATION_LOAD) / (
        FULL_EXTENSION_LOAD - INNER_PRESERVATION_LOAD
    )
    return float(t**3 * (10.0 - 15.0 * t + 6.0 * t**2))


def _fit_pair_extension(
    coordinates: np.ndarray, residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    centers = []
    even_targets = []
    odd_targets = []
    radii = []
    for negative in range(0, coordinates.shape[0], 2):
        positive = negative + 1
        odd_coordinate = 0.5 * (
            coordinates[positive] - coordinates[negative]
        )
        radius = float(np.linalg.norm(odd_coordinate))
        if radius <= np.finfo(float).tiny:
            raise RuntimeError("shell-extension signed radius vanished")
        centers.append(odd_coordinate / radius)
        radii.append(radius)
        even_targets.append(
            0.5 * (residual[positive] + residual[negative]) / radius**EVEN_POWER
        )
        odd_targets.append(
            0.5 * (residual[positive] - residual[negative]) / radius**ODD_POWER
        )
    centers_array = np.asarray(centers, dtype=float)
    even_system = (centers_array @ centers_array.T) ** EVEN_POWER
    odd_system = (centers_array @ centers_array.T) ** ODD_POWER
    even_regularized = even_system + KERNEL_REGULARIZATION * np.eye(
        centers_array.shape[0]
    )
    odd_regularized = odd_system + KERNEL_REGULARIZATION * np.eye(
        centers_array.shape[0]
    )
    even_coefficients = np.linalg.solve(
        even_regularized, np.asarray(even_targets, dtype=float)
    )
    odd_coefficients = np.linalg.solve(
        odd_regularized, np.asarray(odd_targets, dtype=float)
    )
    metrics = {
        "pair_count": int(centers_array.shape[0]),
        "minimum_training_radius": float(np.min(radii)),
        "maximum_training_radius": float(np.max(radii)),
        "even_kernel_rank": int(np.linalg.matrix_rank(even_regularized)),
        "odd_kernel_rank": int(np.linalg.matrix_rank(odd_regularized)),
        "even_kernel_condition_number": float(np.linalg.cond(even_regularized)),
        "odd_kernel_condition_number": float(np.linalg.cond(odd_regularized)),
    }
    return centers_array, even_coefficients, odd_coefficients, metrics


def _extension_value(
    coordinate: np.ndarray,
    centers: np.ndarray,
    even_coefficients: np.ndarray,
    odd_coefficients: np.ndarray,
) -> np.ndarray:
    point = np.asarray(coordinate, dtype=float)
    radius = float(np.linalg.norm(point))
    if radius <= np.finfo(float).tiny:
        return np.zeros(even_coefficients.shape[1], dtype=float)
    unit = point / radius
    return (
        radius**EVEN_POWER
        * ((unit @ centers.T) ** EVEN_POWER @ even_coefficients)
        + radius**ODD_POWER
        * ((unit @ centers.T) ** ODD_POWER @ odd_coefficients)
    )


def _relative_rows(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(actual) - np.asarray(expected), axis=1) / np.maximum(
        np.linalg.norm(np.asarray(expected), axis=1), np.finfo(float).tiny
    )


def _fit_extension(
    *, include_physical_audits: bool = True
) -> tuple[dict[str, np.ndarray], dict]:
    truth = _load_npz(PARENT_ARRAYS)
    model = vector_field.ReducedVectorField()
    states = np.asarray(truth["candidate_primitive_states"], dtype=float)
    exact_deltas = np.asarray(truth["candidate_scaled_deltas"], dtype=float)
    exact_rates = np.asarray(truth["total_rates_per_second"], dtype=float)
    departures = np.asarray(truth["candidate_departure_coordinates"], dtype=float)
    memories = exact_deltas @ model.memory_basis
    coordinates = np.concatenate(
        (np.zeros((8, 162), dtype=float), memories, departures), axis=1
    )
    old_decoded_deltas = np.asarray(
        [model.decoded_delta(coordinate) for coordinate in coordinates], dtype=float
    )
    departures = coordinates[:, -28:]
    if (
        coordinates.shape != (8, 470)
        or departures.shape != (8, 28)
        or not np.allclose(
            departures,
            truth["candidate_departure_coordinates"],
            rtol=0.0,
            atol=2.0e-11,
        )
    ):
        raise RuntimeError("shell-extension training coordinates changed")
    old_decoded_deltas = np.asarray(old_decoded_deltas, dtype=float)
    decoder_residual = exact_deltas - old_decoded_deltas
    centers, decoder_even, decoder_odd, decoder_fit = _fit_pair_extension(
        departures, decoder_residual
    )
    loads = np.max(np.abs(old_decoded_deltas), axis=1)
    weights = np.asarray([_shell_weight(load) for load in loads])
    if not np.array_equal(weights, np.ones(8)):
        raise RuntimeError("outer-shell training weights changed")
    extended_deltas = np.asarray(
        [
            old_delta
            + weight
            * _extension_value(departure, centers, decoder_even, decoder_odd)
            for old_delta, departure, weight in zip(
                old_decoded_deltas, departures, weights
            )
        ]
    )
    baseline_rates = np.asarray(
        [
            model.base_rate
            + model.generator @ delta
            + model.departure_basis @ model.nonlinear_departure(departure)
            for delta, departure in zip(extended_deltas, departures)
        ]
    )
    rate_residual = exact_rates - baseline_rates
    rate_centers, rate_even, rate_odd, rate_fit = _fit_pair_extension(
        departures, rate_residual
    )
    if not np.allclose(rate_centers, centers, rtol=0.0, atol=1.0e-14):
        raise RuntimeError("decoder/rate shell centers changed")
    extended_rates = np.asarray(
        [
            baseline
            + weight * _extension_value(departure, centers, rate_even, rate_odd)
            for baseline, departure, weight in zip(
                baseline_rates, departures, weights
            )
        ]
    )

    inner = _load_npz(OLD_CLOSURE)
    inner_decoded_all = np.asarray(inner["predicted_scaled_deltas"], dtype=float)
    if inner_decoded_all.shape != (48, 560):
        raise RuntimeError("certified inner decoded states changed")
    finite_inner = np.all(np.isfinite(inner_decoded_all), axis=1)
    inner_decoded = inner_decoded_all[finite_inner]
    if inner_decoded.shape != (32, 560):
        raise RuntimeError("certified inner decoder holdout count changed")
    inner_weights = np.asarray(
        [
            _shell_weight(float(np.max(np.abs(delta))))
            for delta in inner_decoded
        ],
        dtype=float,
    )

    state_audits = []
    coordinate_mismatches = []
    if include_physical_audits:
        for target, delta in zip(coordinates, extended_deltas):
            state = model.base_state + (
                model.columns.ravel() * delta
            ).reshape(model.base_state.shape)
            decoded, factors = model.coordinate(state)
            physical = vector_field.manifest.parent.geometry.chart_tools._state_audit(
                model.components["context"], state
            )
            coordinate_mismatches.append(
                float(
                    np.linalg.norm(decoded - target)
                    / max(float(np.linalg.norm(target)), np.finfo(float).tiny)
                )
            )
            state_audits.append(
                {
                    "minimum_reconstruction_factor": min(
                        float(np.min(factors)), physical["minimum_reconstruction_factor"]
                    ),
                    "maximum_H_over_R": physical["maximum_h_over_r"],
                    "minimum_scattering_optical_depth": physical[
                        "minimum_scattering_optical_depth"
                    ],
                }
            )

    old_decoder_errors = _relative_rows(old_decoded_deltas, exact_deltas)
    extended_decoder_errors = _relative_rows(extended_deltas, exact_deltas)
    baseline_rate_errors = _relative_rows(baseline_rates, exact_rates)
    extended_rate_errors = _relative_rows(extended_rates, exact_rates)
    singular_values = np.linalg.svd(rate_residual, compute_uv=False)
    singular_fraction = singular_values**2 / np.sum(singular_values**2)
    metrics = {
        "decoder_fit": decoder_fit,
        "rate_fit": rate_fit,
        "minimum_outer_training_shell_weight": float(np.min(weights)),
        "maximum_inner_certificate_shell_weight": float(np.max(inner_weights)),
        "old_decoder_maximum_relative_error": float(np.max(old_decoder_errors)),
        "extended_decoder_maximum_relative_error": float(
            np.max(extended_decoder_errors)
        ),
        "baseline_full_state_rate_maximum_relative_error": float(
            np.max(baseline_rate_errors)
        ),
        "extended_full_state_rate_maximum_relative_error": float(
            np.max(extended_rate_errors)
        ),
        "extended_full_state_rate_median_relative_error": float(
            np.median(extended_rate_errors)
        ),
        "rate_residual_leading_singular_energy_fraction": float(
            singular_fraction[0]
        ),
        "maximum_extended_decoder_coordinate_relative_mismatch": (
            float(np.max(coordinate_mismatches))
            if coordinate_mismatches
            else math.nan
        ),
        "minimum_extended_state_reconstruction_factor": (
            float(min(item["minimum_reconstruction_factor"] for item in state_audits))
            if state_audits
            else math.nan
        ),
        "maximum_extended_state_H_over_R": (
            float(max(item["maximum_H_over_R"] for item in state_audits))
            if state_audits
            else math.nan
        ),
        "minimum_extended_state_scattering_optical_depth": (
            float(
                min(
                    item["minimum_scattering_optical_depth"]
                    for item in state_audits
                )
            )
            if state_audits
            else math.nan
        ),
        "new_continuous_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
    }
    arrays = {
        "training_online_coordinates": coordinates,
        "training_departure_coordinates": departures,
        "training_old_decoded_deltas": old_decoded_deltas,
        "training_extended_decoded_deltas": extended_deltas,
        "training_exact_scaled_deltas": exact_deltas,
        "training_baseline_full_state_rates_per_second": baseline_rates,
        "training_extended_full_state_rates_per_second": extended_rates,
        "training_exact_full_state_rates_per_second": exact_rates,
        "training_shell_weights": weights,
        "extension_center_directions": centers,
        "decoder_even4_coefficients": decoder_even,
        "decoder_odd5_coefficients": decoder_odd,
        "full_state_rate_even4_coefficients": rate_even,
        "full_state_rate_odd5_coefficients": rate_odd,
        "rate_residual_singular_values": singular_values,
        "inner_certificate_shell_weights": inner_weights,
    }
    return arrays, metrics


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("holdout direction vanished")
    return array / norms[:, None]


def _holdout_design() -> tuple[np.ndarray, tuple[str, ...], dict]:
    source = _load_npz(DIRECTION_DESIGN)
    directions = np.asarray(source["directions"], dtype=float)
    energy6 = directions[6]
    escape = directions[8]
    forward = directions[9]
    holdouts = _normalize_rows(
        np.vstack(
            (
                forward + HOLDOUT_MIXING * escape,
                forward - HOLDOUT_MIXING * escape,
                forward + HOLDOUT_MIXING * energy6,
                forward - HOLDOUT_MIXING * energy6,
            )
        )
    )
    labels = (
        "forward_plus_0p25_escape",
        "forward_minus_0p25_escape",
        "forward_plus_0p25_energy6",
        "forward_minus_0p25_energy6",
    )
    if holdouts.shape != (HOLDOUT_DIRECTION_COUNT, 28):
        raise RuntimeError("mixed-direction holdout design changed")
    training = directions[[4, 6, 8, 9]]
    return holdouts, labels, {
        "maximum_holdout_to_training_absolute_cosine": float(
            np.max(np.abs(holdouts @ training.T))
        ),
        "minimum_holdout_pair_separation": float(
            np.min(
                np.linalg.norm(
                    holdouts[:, None, :] - holdouts[None, :, :]
                    + np.eye(HOLDOUT_DIRECTION_COUNT)[:, :, None] * 1.0e6,
                    axis=2,
                )
            )
        ),
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "local_extension": {
            "input": "a28_departure_coordinate",
            "decoder_output": "scaled_full_state_delta_560",
            "rate_output": "scaled_full_state_rate_560_per_second",
            "even_homogeneous_power": EVEN_POWER,
            "odd_homogeneous_power": ODD_POWER,
            "kernel_regularization": KERNEL_REGULARIZATION,
            "training_signed_pairs": 4,
            "model_refit_after_holdout_truth": False,
        },
        "shell_gate": {
            "load_definition": "linf_of_old_decoded_scaled_state_delta",
            "identically_old_model_at_or_below": INNER_PRESERVATION_LOAD,
            "fully_extended_model_at_or_above": FULL_EXTENSION_LOAD,
            "transition": "C2_quintic_smootherstep",
            "recenter_trigger": RECENTER_TRIGGER_LOAD,
            "hard_chart_limit": HARD_CHART_LOAD,
        },
        "binding_revealed_fit_gates": {
            "maximum_even_kernel_condition_number": 2.0,
            "maximum_odd_kernel_condition_number": 2.0,
            "minimum_outer_training_shell_weight": 1.0,
            "maximum_inner_certificate_shell_weight": 0.0,
            "maximum_extended_decoder_relative_error": 1.0e-3,
            "maximum_extended_full_state_rate_relative_error": 5.0e-3,
            "maximum_extended_decoder_coordinate_relative_mismatch": 2.5e-3,
            "minimum_extended_state_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_extended_state_H_over_R": 0.12,
            "minimum_extended_state_scattering_optical_depth": 1.0,
        },
        "independent_holdout_geometry": {
            "component_bounds": list(HOLDOUT_COMPONENT_BOUNDS),
            "rung_order": "strictly_increasing_fail_fast",
            "positive_forward_sector_only": True,
            "mixing_coefficient": HOLDOUT_MIXING,
            "direction_count": HOLDOUT_DIRECTION_COUNT,
            "planned_candidate_count": PLANNED_GEOMETRY_CANDIDATES,
            "new_rate_calls_during_geometry_equal": 0,
        },
        "future_independent_rate_gates": {
            "maximum_full_state_rate_relative_error": 0.15,
            "median_full_state_rate_relative_error": 0.075,
            "maximum_a28_rate_relative_error": 0.15,
            "radial_sign_disagreement_count_equal": 0,
            "maximum_decoder_full_state_relative_error": 5.0e-3,
            "maximum_decoder_coordinate_relative_mismatch": 5.0e-3,
        },
        "atlas_transition": {
            "center_source": "accepted_authentic_propagated_state_only",
            "geometry_only_holdout_may_become_center": False,
            "recenter_before_hard_chart_limit": True,
            "coordinate_transition_requires_roundtrip_and_rate_parity": True,
            "stable_memory_transport": "exact_basis_transform_not_algebraic_elimination",
        },
        "target_cycle_architecture": {
            "q162": "conservative_active_coordinates_with_multi_anchor_flux",
            "z280": "dynamic_exponential_or_L_stable_memory_update",
            "a28": "shell_gated_local_transient_atlas_then_phase_or_invariant_measure_closure",
            "online_truth_calls_per_macrostep": 0,
            "maximum_macrosteps_per_cycle": 100_000,
        },
        "authorization_boundaries": {
            "new_truth_calls": 0,
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
        "decoder_even_condition": metrics["decoder_fit"][
            "even_kernel_condition_number"
        ] <= gates["maximum_even_kernel_condition_number"],
        "rate_even_condition": metrics["rate_fit"][
            "even_kernel_condition_number"
        ] <= gates["maximum_even_kernel_condition_number"],
        "decoder_odd_condition": metrics["decoder_fit"][
            "odd_kernel_condition_number"
        ] <= gates["maximum_odd_kernel_condition_number"],
        "rate_odd_condition": metrics["rate_fit"][
            "odd_kernel_condition_number"
        ] <= gates["maximum_odd_kernel_condition_number"],
        "outer_weight": metrics["minimum_outer_training_shell_weight"]
        >= gates["minimum_outer_training_shell_weight"],
        "inner_preservation": metrics["maximum_inner_certificate_shell_weight"]
        <= gates["maximum_inner_certificate_shell_weight"],
        "decoder_error": metrics["extended_decoder_maximum_relative_error"]
        <= gates["maximum_extended_decoder_relative_error"],
        "rate_error": metrics["extended_full_state_rate_maximum_relative_error"]
        <= gates["maximum_extended_full_state_rate_relative_error"],
        "coordinate_mismatch": metrics[
            "maximum_extended_decoder_coordinate_relative_mismatch"
        ] <= gates["maximum_extended_decoder_coordinate_relative_mismatch"],
        "reconstruction": metrics["minimum_extended_state_reconstruction_factor"]
        >= gates["minimum_extended_state_reconstruction_factor"],
        "height": metrics["maximum_extended_state_H_over_R"]
        <= gates["maximum_extended_state_H_over_R"],
        "optical_depth": metrics[
            "minimum_extended_state_scattering_optical_depth"
        ] >= gates["minimum_extended_state_scattering_optical_depth"],
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
        raise RuntimeError("shell-gated atlas manifest already canonicalized")
    arrays, metrics = _fit_extension()
    holdouts, labels, holdout_metrics = _holdout_design()
    contract = _contract()
    checks = _fit_checks(metrics, contract["binding_revealed_fit_gates"])
    if not all(checks.values()):
        raise RuntimeError(f"revealed shell-extension design failed: {checks}")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        CANONICAL_DIRECTORY / "local_atlas_extension.npz", **arrays
    )
    np.savez_compressed(
        CANONICAL_DIRECTORY / "holdout_design.npz",
        directions=holdouts,
        component_bounds=np.asarray(HOLDOUT_COMPONENT_BOUNDS),
    )
    _write_json(
        CANONICAL_DIRECTORY / "design_metrics.json",
        {"checks": checks, "fit": metrics, "holdout": holdout_metrics},
    )
    _write_json(
        CANONICAL_DIRECTORY / "holdout_design.json", {"labels": labels}
    )
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "parent_arrays_sha256": _sha(PARENT_ARRAYS),
            "old_closure_sha256": _sha(OLD_CLOSURE),
            "direction_design_sha256": _sha(DIRECTION_DESIGN),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "old_inner_certificate_preserved_exactly": True,
        "extended_decoder_maximum_training_relative_error": metrics[
            "extended_decoder_maximum_relative_error"
        ],
        "extended_full_state_rate_maximum_training_relative_error": metrics[
            "extended_full_state_rate_maximum_relative_error"
        ],
        "planned_independent_geometry_candidates": PLANNED_GEOMETRY_CANDIDATES,
        "new_truth_calls": 0,
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
        vector_field.THIS_RUNNER,
        vector_field.THIS_TEST,
        direction_source.THIS_RUNNER,
        direction_source.THIS_TEST,
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
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Shell-gated recentered-atlas manifest WP10c9d6c7c3b5c4f25cg",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "A minimal degree-4 even / degree-5 odd correction is frozen for both the 560D decoder and 560D state-space rate. It is driven only by the 28D departure coordinate.",
                "",
                f"The correction is identically zero through load `{INNER_PRESERVATION_LOAD:.3f}`, reaches full weight at `{FULL_EXTENSION_LOAD:.3f}`, triggers recentering at `{RECENTER_TRIGGER_LOAD:.3f}`, and retains `{HARD_CHART_LOAD:.3f}` as a hard limit. Thus all prior inner-chart predictions are bitwise unchanged.",
                "",
                f"Revealed training maximum decoder/rate errors are `{metrics['extended_decoder_maximum_relative_error']:.6e}` / `{metrics['extended_full_state_rate_maximum_relative_error']:.6e}`. These are fit diagnostics, not independent validation.",
                "",
                f"The only authorized next artifact is `{AUTHORIZED_NEXT}`: geometry-first validation of eight prospectively frozen mixed forward-sector holdouts. No new truth call or trajectory was made.",
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
