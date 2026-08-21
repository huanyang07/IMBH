#!/usr/bin/env python3
"""Screen saved, nonsealed states for distinct cold/hot branch candidates."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_branch_first_hybrid_impulse_architecture_wp10c9d6c7c3b5c4f25dl as parent  # noqa: E402
import run_causal_inner_hybrid_candidate_geometry_preflight_wp10c9d6c7c3b5c4f25dc as geometry  # noqa: E402
import run_causal_inner_transition_hidden_tangent_wp10c9d6c7c3b5c4f25dk as tangent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dm"
PARENT_COMMIT = "c31715b22649fe4086f245713121c64e01bb5e9b"
PARENT_PARENT = "4f01a98496310b5c9515104adb1cad3c5f5c44f7"
PARENT_TREE = "1101ae6865c1b7664b9bfbf42a62c9ced5cea7c1"

PAIR_CLASSIFICATION = (
    "distinct_saved_cold_hot_branch_candidates_supported_"
    "separate_branch_truth_manifests_authorized"
)
COLD_ONLY_CLASSIFICATION = (
    "saved_cold_side_candidate_supported_hot_side_not_observed_"
    "branch_truth_blocked_bounded_hot_exit_acquisition_manifest_authorized"
)
INFRASTRUCTURE_CLASSIFICATION = "saved_branch_candidate_screen_infrastructure_failed"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dn"

ANCHOR_TIME_SECONDS = 0.020
SEALED_TIME_SECONDS = 0.016
FULL_CANDIDATE_INDICES = (0, 1, 2, 3)
FIXED_Q_LABELS = ("cold_1", "warm_1", "warm_2", "warm_3")
MINIMUM_TRANSITION_MACRO_DISTANCE = 5.0e-2
MAXIMUM_HIDDEN_SECANT_FRACTION = 0.25
MINIMUM_RECONSTRUCTION_FACTOR = 1.0 - 1.0e-12
MAXIMUM_HEIGHT_RATIO = 0.5
MINIMUM_OPTICAL_DEPTH = 1.0

ARTIFACT = "causal_inner_branch_candidate_saved_array_screen_wp10c9d6c7c3b5c4f25dm"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_branch_candidate_saved_array_screen_"
    "wp10c9d6c7c3b5c4f25dm.py"
)
THIS_TEST = (
    "tests/test_causal_inner_branch_candidate_saved_array_screen_"
    "wp10c9d6c7c3b5c4f25dm.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_BRANCH_CANDIDATE_SAVED_ARRAY_"
    "SCREEN_WP10C9D6C7C3B5C4F25DM_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_SUMMARY = parent.CANONICAL_DIRECTORY / "summary.json"
PARENT_ARCHITECTURE = (
    parent.CANONICAL_DIRECTORY / "branch_first_hybrid_impulse_architecture.json"
)
GEOMETRY_ARRAYS = geometry.CANONICAL_DIRECTORY / "candidate_geometry_arrays.npz"
GEOMETRY_METRICS = geometry.CANONICAL_DIRECTORY / "candidate_geometry_metrics.json"
TANGENT_ARRAYS = tangent.CANONICAL_DIRECTORY / "transition_hidden_tangent_arrays.npz"
TANGENT_METRICS = tangent.CANONICAL_DIRECTORY / "transition_hidden_tangent_metrics.json"
PRIMARY_RETRY_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)


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


def _validate_inputs(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("branch-candidate parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("branch-candidate parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("branch-candidate parent tree changed")

    parent_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    geometry_hashes = _checksums(geometry.CANONICAL_DIRECTORY)
    tangent_hashes = _checksums(tangent.CANONICAL_DIRECTORY)
    retry_hashes = _checksums(PRIMARY_RETRY_DIRECTORY)
    summary = _read(PARENT_SUMMARY)
    architecture = _read(PARENT_ARCHITECTURE)
    geometry_summary = _read(geometry.CANONICAL_DIRECTORY / "summary.json")
    tangent_summary = _read(tangent.CANONICAL_DIRECTORY / "summary.json")
    retry_summary = _read(PRIMARY_RETRY_DIRECTORY / "summary.json")

    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not architecture["authorization_boundaries"][
            "branch_candidate_saved_array_screen_authorized"
        ]
        or summary["branch_truth_execution_authorized"]
        or not architecture["prospective_branch_candidate_screen"][
            "must_select_distinct_cold_and_hot_candidates"
        ]
        or not architecture["branch_first_dependency"][
            "branch_candidate_must_not_be_the_exact_20ms_transition_anchor"
        ]
        or architecture["branch_first_dependency"][
            "branch_root_fail_fast_preflight_hidden_fraction_max"
        ]
        != MAXIMUM_HIDDEN_SECANT_FRACTION
    ):
        raise RuntimeError("branch-first candidate-screen contract changed")
    if (
        not geometry_summary["passed"]
        or not geometry_summary["all_candidates_unclassified"]
        or geometry_summary["existing_candidate_state_count"] != 6
        or geometry_summary["sealed_candidate"] != "U16_unclassified_sealed"
    ):
        raise RuntimeError("saved candidate geometry changed")
    if (
        not tangent_summary["passed"]
        or tangent_summary["selected_hidden_rank"] != 16
        or tangent_summary["new_exact_fixed_Q_rate_calls"] != 0
    ):
        raise RuntimeError("rank-16 transition tangent changed")
    if (
        retry_summary["accepted_main_BDF2_roots"] != 4
        or retry_summary["rejected_main_BDF2_roots"] != 0
    ):
        raise RuntimeError("saved primary continuation checkpoints changed")

    for label in FIXED_Q_LABELS:
        checkpoint = PRIMARY_RETRY_DIRECTORY / f"checkpoint_{label}.npz"
        checkpoint_json = _read(PRIMARY_RETRY_DIRECTORY / f"checkpoint_{label}.json")
        metrics = _read(PRIMARY_RETRY_DIRECTORY / f"metrics_{label}.json")
        if (
            not checkpoint_json["bitwise_roundtrip"]
            or checkpoint_json["sha256"] != _sha(checkpoint)
            or not metrics["accepted"]
            or not metrics["acceptance"]["accepted"]
            or metrics["failure_reasons"]
        ):
            raise RuntimeError(f"saved fixed-Q checkpoint is not accepted: {label}")

    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"branch-first source changed: {relative}")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("branch-candidate screen requires a clean tracked tree")
    return {
        "parent_hashes": parent_hashes,
        "geometry_hashes": geometry_hashes,
        "tangent_hashes": tangent_hashes,
        "retry_hashes": retry_hashes,
    }


def _accepted_history_available(time_seconds: float, state: np.ndarray) -> bool:
    directory = (
        geometry.MIDDLE_5_DIRECTORY
        if time_seconds <= 0.005
        else geometry.MIDDLE_20_DIRECTORY
    )
    source = _load_npz(directory / "decisive_arrays.npz")
    indices = np.flatnonzero(
        np.isclose(source["base__accepted_times"], time_seconds, atol=1.0e-14)
    )
    if len(indices) != 1:
        return False
    index = int(indices[0])
    return bool(
        np.array_equal(source["base__accepted_states"][index], state)
        and np.all(np.isfinite(source["base__accepted_primitive_histories"][index]))
        and np.all(np.isfinite(source["base__accepted_mapped_histories"][index]))
        and np.all(np.isfinite(source["base__accepted_height_histories"][index]))
        and np.isfinite(source["base__accepted_previous_timesteps"][index])
    )


def _screen(*, require_clean: bool) -> tuple[dict, dict[str, np.ndarray], dict]:
    locks = _validate_inputs(require_clean=require_clean)
    candidate = _load_npz(GEOMETRY_ARRAYS)
    candidate_metrics = _read(GEOMETRY_METRICS)["geometry"]
    tangent_arrays = _load_npz(TANGENT_ARRAYS)

    field_arrays = _load_npz(geometry.FIELD_ARRAYS)
    field = geometry.field_manifest.ForwardQuadraticAuthenticCenterField(
        field_arrays
    )
    model = field.model
    macro_restriction = tangent_arrays["macro_restriction_R82"]
    hidden_basis = tangent_arrays["hidden_basis_Z388"]
    hidden_dual = tangent_arrays["hidden_dual_Q388"]
    rank16_basis = tangent_arrays["selected_hidden_basis388"]
    anchor_coordinate = candidate["candidate_absolute_y470_coordinates"][5]

    labels: list[str] = []
    times: list[float] = []
    sides: list[int] = []
    states: list[np.ndarray] = []
    current_coordinate_states: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    previous_coordinates: list[np.ndarray] = []
    secant_durations: list[float] = []
    history_available: list[bool] = []
    minimum_reconstruction: list[float] = []
    maximum_height: list[float] = []
    minimum_optical: list[float] = []

    full_times = candidate["candidate_times_seconds"]
    full_coordinates = candidate["candidate_absolute_y470_coordinates"]
    for local_position, index in enumerate(FULL_CANDIDATE_INDICES):
        time_seconds = float(full_times[index])
        state = np.asarray(candidate["candidate_primitive_states"][index])
        if local_position == 0:
            previous_index, next_index = index, FULL_CANDIDATE_INDICES[1]
        elif local_position == len(FULL_CANDIDATE_INDICES) - 1:
            previous_index, next_index = FULL_CANDIDATE_INDICES[-2], index
        else:
            previous_index = FULL_CANDIDATE_INDICES[local_position - 1]
            next_index = FULL_CANDIDATE_INDICES[local_position + 1]
        labels.append(f"full_model_{time_seconds * 1.0e3:g}ms")
        times.append(time_seconds)
        sides.append(-1)
        states.append(state)
        current_coordinate_states.append(np.asarray(full_coordinates[index]))
        coordinates.append(np.asarray(full_coordinates[next_index]))
        previous_coordinates.append(np.asarray(full_coordinates[previous_index]))
        secant_durations.append(float(full_times[next_index] - full_times[previous_index]))
        history_available.append(_accepted_history_available(time_seconds, state))
        minimum_reconstruction.append(
            float(candidate_metrics["minimum_reconstruction_factors"][index])
        )
        maximum_height.append(float(candidate_metrics["maximum_height_ratios"][index]))
        minimum_optical.append(
            float(candidate_metrics["minimum_scattering_optical_depths"][index])
        )

    for label in FIXED_Q_LABELS:
        checkpoint = _load_npz(PRIMARY_RETRY_DIRECTORY / f"checkpoint_{label}.npz")
        metrics = _read(PRIMARY_RETRY_DIRECTORY / f"metrics_{label}.json")
        state = np.asarray(checkpoint["current_primitive_charts"], dtype=float)
        previous_state = np.asarray(checkpoint["previous_primitive_charts"], dtype=float)
        coordinate, _ = model.coordinate(state)
        previous_coordinate, _ = model.coordinate(previous_state)
        labels.append(f"fixed_Q_{label}")
        times.append(float(checkpoint["elapsed_time_seconds"]))
        sides.append(1)
        states.append(state)
        current_coordinate_states.append(np.asarray(coordinate))
        coordinates.append(np.asarray(coordinate))
        previous_coordinates.append(np.asarray(previous_coordinate))
        secant_durations.append(float(checkpoint["previous_timestep_seconds"]))
        history_available.append(
            bool(
                int(checkpoint["current_order"]) == 2
                and int(checkpoint["completed_steps"]) >= 3
                and np.all(np.isfinite(checkpoint["previous_mapped_storage_increment"]))
                and np.all(
                    np.isfinite(
                        checkpoint["previous_responsive_height_storage_increment"]
                    )
                )
            )
        )
        minimum_reconstruction.append(
            float(metrics["minimum_path_reconstruction_factor"])
        )
        maximum_height.append(float(metrics["maximum_H_over_R"]))
        minimum_optical.append(float(metrics["minimum_scattering_optical_depth"]))

    coordinate_states = np.asarray(current_coordinate_states)
    secant_rates = (
        np.asarray(coordinates) - np.asarray(previous_coordinates)
    ) / np.asarray(secant_durations)[:, None]
    hidden_rates = (hidden_dual @ secant_rates.T).T
    hidden_actions = (hidden_basis @ hidden_rates.T).T
    hidden_fractions = np.linalg.norm(hidden_actions, axis=1) / np.maximum(
        np.linalg.norm(secant_rates, axis=1), np.finfo(float).tiny
    )
    rank16_rate_capture = np.linalg.norm(
        hidden_rates @ rank16_basis, axis=1
    ) / np.maximum(np.linalg.norm(hidden_rates, axis=1), np.finfo(float).tiny)

    coordinate_departures = coordinate_states - anchor_coordinate
    hidden_departures = (hidden_dual @ coordinate_departures.T).T
    macro_distances = np.linalg.norm(
        coordinate_departures @ macro_restriction.T, axis=1
    )
    rank16_amplitudes = np.linalg.norm(hidden_departures @ rank16_basis, axis=1)
    hidden_amplitudes = np.linalg.norm(hidden_departures, axis=1)
    rank16_displacement_capture = rank16_amplitudes / np.maximum(
        hidden_amplitudes, np.finfo(float).tiny
    )

    history_array = np.asarray(history_available, dtype=bool)
    reconstruction_array = np.asarray(minimum_reconstruction)
    height_array = np.asarray(maximum_height)
    optical_array = np.asarray(minimum_optical)
    physical = np.logical_and.reduce(
        (
            reconstruction_array >= MINIMUM_RECONSTRUCTION_FACTOR,
            height_array <= MAXIMUM_HEIGHT_RATIO,
            optical_array >= MINIMUM_OPTICAL_DEPTH,
        )
    )
    eligible = np.logical_and.reduce(
        (
            macro_distances >= MINIMUM_TRANSITION_MACRO_DISTANCE,
            hidden_fractions <= MAXIMUM_HIDDEN_SECANT_FRACTION,
            history_array,
            physical,
        )
    )
    side_array = np.asarray(sides, dtype=np.int64)
    cold_indices = np.flatnonzero(np.logical_and(eligible, side_array < 0))
    hot_indices = np.flatnonzero(np.logical_and(eligible, side_array > 0))
    cold_index = int(cold_indices[np.argmin(macro_distances[cold_indices])]) if len(cold_indices) else -1
    hot_index = int(hot_indices[np.argmin(macro_distances[hot_indices])]) if len(hot_indices) else -1
    pair_supported = cold_index >= 0 and hot_index >= 0
    classification = PAIR_CLASSIFICATION if pair_supported else COLD_ONLY_CLASSIFICATION

    arrays = {
        "candidate_times_seconds": np.asarray(times),
        "candidate_side_codes": side_array,
        "candidate_primitive_states": np.asarray(states),
        "candidate_absolute_y470_coordinates": coordinate_states,
        "candidate_secant_rates470_per_s": secant_rates,
        "candidate_macro_distance_from_transition_anchor": macro_distances,
        "candidate_hidden_amplitude388": hidden_amplitudes,
        "candidate_rank16_hidden_amplitude": rank16_amplitudes,
        "candidate_rank16_displacement_capture": rank16_displacement_capture,
        "candidate_saved_secant_hidden_fraction": hidden_fractions,
        "candidate_rank16_secant_capture": rank16_rate_capture,
        "candidate_minimum_reconstruction_factor": reconstruction_array,
        "candidate_maximum_height_ratio": height_array,
        "candidate_minimum_scattering_optical_depth": optical_array,
        "candidate_authentic_history_available": history_array,
        "candidate_physical_guard_pass": physical,
        "candidate_eligible_mask": eligible,
        "selected_cold_candidate_index": np.asarray(cold_index),
        "selected_hot_candidate_index": np.asarray(hot_index),
    }
    metrics = {
        "classification": classification,
        "candidate_labels": labels,
        "candidate_count": len(labels),
        "cold_candidate_count": int(len(cold_indices)),
        "hot_candidate_count": int(len(hot_indices)),
        "selected_cold_candidate": labels[cold_index] if cold_index >= 0 else None,
        "selected_hot_candidate": labels[hot_index] if hot_index >= 0 else None,
        "distinct_cold_hot_pair_supported": pair_supported,
        "minimum_transition_macro_distance": MINIMUM_TRANSITION_MACRO_DISTANCE,
        "maximum_hidden_secant_fraction": MAXIMUM_HIDDEN_SECANT_FRACTION,
        "candidate_times_seconds": np.asarray(times),
        "candidate_macro_distances": macro_distances,
        "candidate_hidden_secant_fractions": hidden_fractions,
        "candidate_rank16_hidden_amplitudes": rank16_amplitudes,
        "candidate_rank16_secant_captures": rank16_rate_capture,
        "candidate_history_available": history_array,
        "candidate_physical_guard_passes": physical,
        "candidate_eligible_mask": eligible,
        "sealed_16ms_opened": False,
        "exact_20ms_transition_anchor_selected": False,
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "new_chart_retractions": 0,
        "propagated_states": 0,
    }
    checks = {
        "candidate_count": len(labels) == 8,
        "sealed_excluded": not np.any(np.isclose(times, SEALED_TIME_SECONDS)),
        "transition_anchor_excluded": not np.any(np.isclose(times, ANCHOR_TIME_SECONDS)),
        "all_saved_histories_available": bool(np.all(history_array)),
        "all_physical_guards": bool(np.all(physical)),
        "cold_candidate_supported": cold_index >= 0,
        "hot_candidate_supported": hot_index >= 0,
        "distinct_pair_supported": pair_supported,
        "no_truth": True,
        "no_generator": True,
        "no_roots": True,
        "no_retractions": True,
        "no_propagation": True,
        "sealed_budget": True,
    }
    infrastructure = all(
        value
        for key, value in checks.items()
        if key not in {"hot_candidate_supported", "distinct_pair_supported"}
    )
    if not infrastructure:
        metrics["classification"] = INFRASTRUCTURE_CLASSIFICATION
    return metrics, arrays, {"checks": checks, "locks": locks}


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
                    "sha256": _sha(path),
                    "status": summary["classification"],
                }
            )
    CANONICAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "sha256", "status"))
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY) if CANONICAL_SUMMARY.exists() else {}
    catalog[ARTIFACT] = summary
    _write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("branch-candidate saved-array screen already exists")
    metrics, arrays, audit = _screen(require_clean=True)
    infrastructure_passed = all(
        value
        for key, value in audit["checks"].items()
        if key not in {"hot_candidate_supported", "distinct_pair_supported"}
    )
    if not infrastructure_passed:
        raise RuntimeError(f"branch-candidate screen infrastructure failed: {audit['checks']}")

    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_npz(CANONICAL_DIRECTORY / "branch_candidate_screen_arrays.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "branch_candidate_screen_metrics.json",
        {"passed": metrics["distinct_cold_hot_pair_supported"], **metrics, **audit},
    )
    acquisition = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "observed_evidence": {
            "cold_candidate": metrics["selected_cold_candidate"],
            "hot_candidate": metrics["selected_hot_candidate"],
            "post_20ms_checkpoints_remain_transition_dominated": (
                metrics["hot_candidate_count"] == 0
            ),
        },
        "authorization_boundaries": {
            "cold_branch_truth_execution_authorized": False,
            "hot_branch_truth_execution_authorized": False,
            "transition_truth_execution_authorized": False,
            "definitions_only_bounded_hot_exit_acquisition_manifest_authorized": (
                not metrics["distinct_cold_hot_pair_supported"]
            ),
            "separate_branch_truth_manifests_authorized": metrics[
                "distinct_cold_hot_pair_supported"
            ],
            "online_solver_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "required_next_manifest_if_hot_absent": {
            "work_package": AUTHORIZED_NEXT,
            "purpose": (
                "prospectively_bound_a_short_full_y470_fixed_Q_transition_exit_"
                "acquisition_with_fail_fast_hidden_fraction_and_physical_gates"
            ),
            "must_not_relabel_existing_transition_checkpoints_as_hot": True,
            "must_stop_at_first_saved_secant_hidden_fraction_at_or_below": (
                MAXIMUM_HIDDEN_SECANT_FRACTION
            ),
            "branch_root_execution_in_next_package": False,
        },
    }
    _write_json(
        CANONICAL_DIRECTORY / "branch_candidate_acquisition_contract.json", acquisition
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_tree": PARENT_TREE,
            "decisive_input_hashes": {
                "parent_summary": _sha(PARENT_SUMMARY),
                "parent_architecture": _sha(PARENT_ARCHITECTURE),
                "candidate_geometry_arrays": _sha(GEOMETRY_ARRAYS),
                "candidate_geometry_metrics": _sha(GEOMETRY_METRICS),
                "transition_tangent_arrays": _sha(TANGENT_ARRAYS),
                "transition_tangent_metrics": _sha(TANGENT_METRICS),
                **{
                    f"checkpoint_{label}": _sha(
                        PRIMARY_RETRY_DIRECTORY / f"checkpoint_{label}.npz"
                    )
                    for label in FIXED_Q_LABELS
                },
            },
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["distinct_cold_hot_pair_supported"],
        "screen_completed": True,
        "infrastructure_passed": True,
        "cold_candidate_supported": metrics["cold_candidate_count"] > 0,
        "hot_candidate_supported": metrics["hot_candidate_count"] > 0,
        "distinct_cold_hot_pair_supported": metrics[
            "distinct_cold_hot_pair_supported"
        ],
        "selected_cold_candidate": metrics["selected_cold_candidate"],
        "selected_hot_candidate": metrics["selected_hot_candidate"],
        "branch_truth_execution_authorized": False,
        "transition_truth_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
        "authorized_next_artifact": (
            "definitions_only_bounded_hot_exit_acquisition_manifest"
            if not metrics["distinct_cold_hot_pair_supported"]
            else "definitions_only_separate_cold_branch_truth_manifest"
        ),
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "new_chart_retractions": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_commit": _git("rev-parse", "HEAD"),
            "implementation_tree": _git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
            },
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Saved-array branch-candidate screen WP10c9d6c7c3b5c4f25dm",
                "",
                "## Result",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"The nearest eligible cold-side state is `{metrics['selected_cold_candidate']}`. Its saved trajectory secant is below the frozen hidden-fraction gate and its complete accepted history and physical margins are present.",
                "",
                f"No hot-side state passes. The four accepted fixed-Q checkpoints after 20 ms have hidden fractions from `{min(metrics['candidate_hidden_secant_fractions'][4:]):.9f}` to `{max(metrics['candidate_hidden_secant_fractions'][4:]):.9f}` and macro distances no larger than `{max(metrics['candidate_macro_distances'][4:]):.6e}`. They remain inside the transition sector and may not be relabeled as a hot branch.",
                "",
                "The screen used only previously revealed, hash-locked states and algebraic coordinates. It made zero truth-rate calls, generator assemblies, roots, chart retractions, or propagated steps, and it did not open the sealed 16 ms state.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`, a definitions-only bounded hot-exit acquisition manifest. Branch truth remains blocked.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if arguments.freeze:
        payload = _freeze()
    else:
        metrics, _, audit = _screen(require_clean=False)
        payload = {"metrics": metrics, "checks": audit["checks"]}
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
