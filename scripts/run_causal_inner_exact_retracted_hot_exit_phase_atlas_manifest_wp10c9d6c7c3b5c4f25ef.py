#!/usr/bin/env python3
"""Freeze the moving exact-chart recovery of the adaptive hot-exit atlas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_adaptive_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25ee as rejected  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ef"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25f0"
PARENT_COMMIT = "d88277077fc2772e91f8f358011d450dc59d25d7"
PARENT_TREE = "163f9e3cfd06bc2fefa4d851fdcec50427d9bfb9"
CLASSIFICATION = (
    "moving_exact_chart_hot_exit_recovery_manifest_frozen_"
    "last_accepted_endpoint_preserved"
)

FIRST_WINDOW_INDEX = 3
MAXIMUM_WINDOW_INDEX = 8
NODE_COUNT = 8
INITIAL_DURATION_SECONDS = 2.0e-7
MAXIMUM_DURATION_SECONDS = 4.0e-7
RATE_BASIS_RANKS = (4, 6, 8, 12, 16)
MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW = 15

MAXIMUM_TRAINING_NORMAL_RATE_DEFECT = 1.0e-4
MAXIMUM_PROJECTED_COLLOCATION_DEFECT = 5.0e-2
MAXIMUM_FULL_COLLOCATION_DEFECT = 5.0e-2
MAXIMUM_NORMAL_RATE_DEFECT = 1.0e-2
MINIMUM_RATE_DIRECTION_COSINE = 0.995
MAXIMUM_Q3_RELATIVE_DRIFT = 5.0e-4
MINIMUM_RECONSTRUCTION_FACTOR = 1.0 - 1.0e-12

COORDINATE_TOLERANCE = 1.0e-10
GAUGE_TOLERANCE = 1.0e-10
MAXIMUM_AUGMENTED_CONDITION_NUMBER = 1.0e7
MAXIMUM_SCALED_ANCHOR_DEPARTURE = 5.0e-2
MAXIMUM_NEWTON_CORRECTIONS = 4

GROW_MAXIMUM_FULL_COLLOCATION_DEFECT = 1.0e-2
GROW_MAXIMUM_NORMAL_RATE_DEFECT = 5.0e-3
GROW_MINIMUM_RATE_DIRECTION_COSINE = 0.999
GROW_MAXIMUM_Q3_RELATIVE_DRIFT = 2.5e-4

HIDDEN_SECANT_FRACTION_MAX = rejected.manifest.HIDDEN_SECANT_FRACTION_MAX
HIDDEN_EXIT_PERSISTENCE_WINDOWS = rejected.manifest.HIDDEN_EXIT_PERSISTENCE_WINDOWS
RANK16_HIDDEN_AMPLITUDE_MIN = rejected.manifest.RANK16_HIDDEN_AMPLITUDE_MIN
MAXIMUM_MACRO_DRIFT_FROM_SEED = rejected.manifest.MAXIMUM_MACRO_DRIFT_FROM_SEED

ARTIFACT = (
    "causal_inner_exact_retracted_hot_exit_phase_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25ef"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
ACCEPTED_DIRECTORY = rejected._stage_directory(2)
REJECTED_DIRECTORY = rejected._stage_directory(3)
THIS_RUNNER = (
    "scripts/run_causal_inner_exact_retracted_hot_exit_phase_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25ef.py"
)
THIS_TEST = (
    "tests/test_causal_inner_exact_retracted_hot_exit_phase_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25ef.py"
)
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_exact_retracted_hot_exit_phase_atlas_"
    "wp10c9d6c7c3b5c4f25f0.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_exact_retracted_hot_exit_phase_atlas_"
    "wp10c9d6c7c3b5c4f25f0.py"
)
EXACT_CHART_SOURCE = (
    "scripts/run_causal_inner_exact_geometric_470_chart_preflight_"
    "wp10c9d6c7c3b5c4f25de.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXACT_RETRACTED_HOT_EXIT_"
    "PHASE_ATLAS_MANIFEST_WP10C9D6C7C3B5C4F25EF_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return rejected._helper()


def _decisive_inputs() -> dict[str, Path]:
    return {
        "adaptive_manifest_summary": rejected.manifest.CANONICAL_DIRECTORY / "summary.json",
        "accepted_window_02_summary": ACCEPTED_DIRECTORY / "summary.json",
        "accepted_window_02_metrics": ACCEPTED_DIRECTORY / "phase_window_metrics.json",
        "accepted_window_02_arrays": ACCEPTED_DIRECTORY / "phase_window_arrays.npz",
        "accepted_window_02_checkpoint": ACCEPTED_DIRECTORY / "phase_window_checkpoint.npz",
        "rejected_window_03_summary": REJECTED_DIRECTORY / "summary.json",
        "rejected_window_03_metrics": REJECTED_DIRECTORY / "phase_window_metrics.json",
        "rejected_window_03_arrays": REJECTED_DIRECTORY / "phase_window_arrays.npz",
    }


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("moving-chart parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("moving-chart parent tree changed")
    accepted_hashes = helper._validate_checksums(ACCEPTED_DIRECTORY)
    rejected_hashes = helper._validate_checksums(REJECTED_DIRECTORY)
    accepted = helper._read(ACCEPTED_DIRECTORY / "summary.json")
    failed = helper._read(REJECTED_DIRECTORY / "summary.json")
    failed_metrics = helper._read(REJECTED_DIRECTORY / "phase_window_metrics.json")
    failed_gates = {name for name, passed in failed_metrics["gates"].items() if not passed}
    if (
        not accepted["passed"]
        or accepted["window_index"] != 2
        or accepted["hot_exit_observed"]
        or failed["passed"]
        or failed["window_index"] != 3
        or failed_gates != {"decoder_roundtrip", "projected_collocation", "full_collocation"}
        or not failed_metrics["gates"]["exact_rate_physics"]
        or not failed_metrics["gates"]["fixed_Q_state_drift"]
    ):
        raise RuntimeError("adaptive chart-failure localization changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("moving-chart manifest requires a clean tracked tree")
    return {
        "accepted_window_02_hashes": accepted_hashes,
        "rejected_window_03_hashes": rejected_hashes,
    }


def _contract(parent: dict) -> dict:
    helper = _helper()
    sources = (
        THIS_RUNNER,
        THIS_TEST,
        EXECUTION_RUNNER,
        EXECUTION_TEST,
        EXACT_CHART_SOURCE,
        rejected.THIS_RUNNER,
        rejected.manifest.original.PHASE_SOURCE,
        rejected.manifest.original.FIXED_Q_SOURCE,
        rejected.manifest.original.EXACT_RATE_SOURCE,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "accepted_seed": str(ACCEPTED_DIRECTORY.relative_to(ROOT)),
        "rejected_diagnostic_only": str(REJECTED_DIRECTORY.relative_to(ROOT)),
        "scientific_interpretation": {
            "last_accepted_endpoint_is_the_only_propagated_state": True,
            "rejected_window_never_seeds_the_recovery": True,
            "failure_localized_to_fixed_affine_decoder_range_and_collocation_window": True,
            "physical_fixed_Q_failure_selected": False,
        },
        "moving_exact_chart": {
            "recenter_at_every_accepted_window": True,
            "anchor_coordinate_is_coordinate_of_accepted_anchor_state": True,
            "gauge_is_canonical_null_basis_at_anchor": True,
            "every_new_truth_state_is_exactly_retracted": True,
            "coordinate_tolerance": COORDINATE_TOLERANCE,
            "gauge_tolerance": GAUGE_TOLERANCE,
            "maximum_augmented_condition_number": MAXIMUM_AUGMENTED_CONDITION_NUMBER,
            "maximum_scaled_anchor_departure": MAXIMUM_SCALED_ANCHOR_DEPARTURE,
            "maximum_Newton_corrections": MAXIMUM_NEWTON_CORRECTIONS,
        },
        "execution": {
            "first_window_index": FIRST_WINDOW_INDEX,
            "maximum_window_index": MAXIMUM_WINDOW_INDEX,
            "initial_duration_seconds": INITIAL_DURATION_SECONDS,
            "maximum_duration_seconds": MAXIMUM_DURATION_SECONDS,
            "node_count": NODE_COUNT,
            "rate_basis_ranks": list(RATE_BASIS_RANKS),
            "maximum_unique_exact_rate_states_per_window": MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW,
            "new_nonlinear_fixed_Q_roots_equal": 0,
            "new_BDF_microsteps_equal": 0,
            "failed_window_never_propagates": True,
        },
        "acceptance_gates": {
            "maximum_training_normal_rate_defect": MAXIMUM_TRAINING_NORMAL_RATE_DEFECT,
            "maximum_projected_collocation_defect": MAXIMUM_PROJECTED_COLLOCATION_DEFECT,
            "maximum_full_collocation_defect": MAXIMUM_FULL_COLLOCATION_DEFECT,
            "maximum_normal_rate_defect": MAXIMUM_NORMAL_RATE_DEFECT,
            "minimum_rate_direction_cosine": MINIMUM_RATE_DIRECTION_COSINE,
            "maximum_Q3_relative_drift": MAXIMUM_Q3_RELATIVE_DRIFT,
            "minimum_reconstruction_factor": MINIMUM_RECONSTRUCTION_FACTOR,
        },
        "event_definition_unchanged": True,
        "authorized_outcomes": [
            "exact_retracted_hot_exit_phase_window_passed_event_not_yet_observed",
            "persistent_hot_exit_candidate_observed_endpoint_refinement_authorized",
            "exact_retracted_hot_exit_phase_atlas_budget_exhausted_without_event",
            "exact_retracted_hot_exit_phase_window_rejected_last_accepted_endpoint_preserved",
        ],
        "parent_hashes": parent,
        "decisive_input_hashes": {
            name: helper._sha(path) for name, path in _decisive_inputs().items()
        },
        "frozen_source_hashes": {
            relative: helper._sha(ROOT / relative) for relative in sources
        },
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = rejected._post().manifest.transition.manifest.cold.manifest
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": helper._sha(path), "scientific_status": "DEFINITIONS_ONLY"})
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": PARENT_COMMIT, "latest_work_package": WORK_PACKAGE})
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("moving-chart recovery manifest already exists")
    parent = _validate_parent(require_clean=True)
    contract = _contract(parent)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "exact_retracted_hot_exit_phase_atlas_contract.json", contract)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", parent)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "last_accepted_window_index": 2,
        "rejected_window_index": 3,
        "exact_retracted_phase_atlas_execution_authorized": True,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {"runner": THIS_RUNNER, "test": THIS_TEST, "implementation_commit": helper._git("rev-parse", "HEAD"), "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"), "python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Moving exact-chart hot-exit recovery manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The accepted atlas remains frozen through Window 2. Rejected Window 3 failed only the fixed affine decoder and collocation gates; exact-rate physics and fixed-Q drift remained admissible. It is diagnostic only and is never propagated.", "", "The recovery recenters a gauge-fixed implicit 470-coordinate chart at every accepted endpoint, exactly retracts every truth state, halves the first retry to 0.2 microseconds, and retains all physical, collocation, event, and fail-closed gates.", "")), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("use --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
