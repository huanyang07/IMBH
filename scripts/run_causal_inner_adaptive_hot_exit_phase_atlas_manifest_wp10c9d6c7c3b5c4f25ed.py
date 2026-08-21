#!/usr/bin/env python3
"""Freeze the adaptive post-transition phase-atlas hot-exit search."""

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

import run_causal_inner_cycle_map_architecture_decision_v2_wp10c9d6c7c3b5c4f25ec as architecture  # noqa: E402
import run_causal_inner_bounded_hot_exit_acquisition_manifest_wp10c9d6c7c3b5c4f25dn as legacy_exit  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ed"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25ee"
PARENT_COMMIT = "7453b93f920d02515305edbf2b2bf0b7d6e8f50b"
PARENT_TREE = "4239a0d09002b8c75e51ad6413bc00e1b2a7e94e"
CLASSIFICATION = "adaptive_hot_exit_rank_adaptive_phase_atlas_manifest_frozen"

NODE_COUNT = 8
INITIAL_DURATION_SECONDS = 2.0e-7
MAXIMUM_DURATION_SECONDS = 6.4e-6
MAXIMUM_WINDOWS = 8
RATE_BASIS_RANKS = (4, 6, 8, 12, 16)
PICARD_UPDATES = 1
MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW = 15

MAXIMUM_TRAINING_NORMAL_RATE_DEFECT = 1.0e-4
MAXIMUM_PROJECTED_COLLOCATION_DEFECT = 5.0e-2
MAXIMUM_FULL_COLLOCATION_DEFECT = 5.0e-2
MAXIMUM_NORMAL_RATE_DEFECT = 1.0e-2
MINIMUM_RATE_DIRECTION_COSINE = 0.995
MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH = 2.0e-3
MAXIMUM_Q3_RELATIVE_DRIFT = 5.0e-4
MINIMUM_RECONSTRUCTION_FACTOR = 1.0 - 1.0e-12

GROW_MAXIMUM_FULL_COLLOCATION_DEFECT = 1.0e-2
GROW_MAXIMUM_NORMAL_RATE_DEFECT = 5.0e-3
GROW_MINIMUM_RATE_DIRECTION_COSINE = 0.999
GROW_MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH = 1.0e-3
GROW_MAXIMUM_Q3_RELATIVE_DRIFT = 2.5e-4

HIDDEN_SECANT_FRACTION_MAX = legacy_exit.HIDDEN_SECANT_FRACTION_MAX
HIDDEN_EXIT_PERSISTENCE_WINDOWS = legacy_exit.HIDDEN_EXIT_PERSISTENCE_STEPS
RANK16_HIDDEN_AMPLITUDE_MIN = legacy_exit.RANK16_HIDDEN_AMPLITUDE_MIN
MAXIMUM_MACRO_DRIFT_FROM_SEED = legacy_exit.MAXIMUM_MACRO_DRIFT_FROM_SEED

ARTIFACT = "causal_inner_adaptive_hot_exit_phase_atlas_manifest_wp10c9d6c7c3b5c4f25ed"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_adaptive_hot_exit_phase_atlas_manifest_wp10c9d6c7c3b5c4f25ed.py"
THIS_TEST = "tests/test_causal_inner_adaptive_hot_exit_phase_atlas_manifest_wp10c9d6c7c3b5c4f25ed.py"
EXECUTION_RUNNER = "scripts/run_causal_inner_adaptive_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25ee.py"
EXECUTION_TEST = "tests/test_causal_inner_adaptive_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25ee.py"
PHASE_SOURCE = "src/imri_qpe/layer3_minidisk_1d/phase_collocation.py"
FIXED_Q_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py"
EXACT_RATE_SOURCE = "scripts/run_causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy.py"
POST_RUNNER = "scripts/run_causal_inner_post_transition_phase_window_wp10c9d6c7c3b5c4f25eb.py"
LEGACY_EXIT_SOURCE = "scripts/run_causal_inner_bounded_hot_exit_acquisition_wp10c9d6c7c3b5c4f25do.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_HOT_EXIT_PHASE_ATLAS_MANIFEST_WP10C9D6C7C3B5C4F25ED_2026-08-21.md"
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return architecture._helper()


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("adaptive phase-atlas parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("adaptive phase-atlas parent tree changed")
    architecture_hashes = helper._validate_checksums(architecture.CANONICAL_DIRECTORY)
    summary = helper._read(architecture.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["working_mathematical_architecture_selected"]
        or not summary["observed_three_mode_prefix_certified"]
        or summary["hot_exit_observed"]
        or summary["authorized_next"] != WORK_PACKAGE
    ):
        raise RuntimeError("cycle-map architecture authorization changed")
    post_hashes = helper._validate_checksums(
        architecture.rejected.post.CANONICAL_DIRECTORY
    )
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("adaptive phase-atlas manifest requires a clean tracked tree")
    return {
        "architecture_hashes": architecture_hashes,
        "post_transition_hashes": post_hashes,
    }


def _decisive_inputs() -> dict[str, Path]:
    post = architecture.rejected.post.CANONICAL_DIRECTORY
    tangent = architecture.rejected.post.manifest.transition.manifest.geometry.manifest
    return {
        "architecture_summary": architecture.CANONICAL_DIRECTORY / "summary.json",
        "architecture_metrics": architecture.CANONICAL_DIRECTORY / "architecture_metrics.json",
        "architecture_arrays": architecture.CANONICAL_DIRECTORY / "architecture_arrays.npz",
        "post_summary": post / "summary.json",
        "post_metrics": post / "post_transition_phase_window_metrics.json",
        "post_witnesses": post / "post_transition_phase_window_model_and_witnesses.npz",
        "hidden_tangent_arrays": tangent.TANGENT_ARRAYS,
        "legacy_hot_exit_contract": legacy_exit.CANONICAL_DIRECTORY / "hot_exit_acquisition_contract.json",
    }


def _contract() -> dict:
    helper = _helper()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "initial_condition": {
            "coordinate": "certified Stage-4 two-half-window endpoint",
            "primitive_state": "certified Stage-4 two-half-window decoded endpoint",
            "fixed_Q_target": "losslessly inherited transition checkpoint target",
            "prior_event_persistence": 0,
        },
        "phase_atlas": {
            "method": "one projected Picard update on eight Legendre-Gauss-Lobatto nodes",
            "one_window_per_committed_stage": True,
            "node_count": NODE_COUNT,
            "initial_duration_seconds": INITIAL_DURATION_SECONDS,
            "maximum_duration_seconds": MAXIMUM_DURATION_SECONDS,
            "maximum_windows": MAXIMUM_WINDOWS,
            "candidate_basis_ranks": RATE_BASIS_RANKS,
            "basis_selection": "minimum candidate rank whose inherited normalized exact-rate witnesses satisfy the training-normal gate",
            "duration_rule": "double after a strong-margin accepted window; hold otherwise; hold after first event-positive window",
            "failed_window_never_propagates": True,
            "no_sequential_BDF_microsteps": True,
        },
        "truth_budget": {
            "maximum_unique_exact_rate_states_per_window": MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW,
            "maximum_unique_exact_rate_states_total": MAXIMUM_WINDOWS * MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW,
            "new_nonlinear_fixed_Q_roots": 0,
            "new_BDF_microsteps": 0,
        },
        "binding_gates": {
            "maximum_training_normal_rate_defect": MAXIMUM_TRAINING_NORMAL_RATE_DEFECT,
            "maximum_projected_collocation_defect": MAXIMUM_PROJECTED_COLLOCATION_DEFECT,
            "maximum_full_collocation_defect": MAXIMUM_FULL_COLLOCATION_DEFECT,
            "maximum_normal_rate_defect": MAXIMUM_NORMAL_RATE_DEFECT,
            "minimum_rate_direction_cosine": MINIMUM_RATE_DIRECTION_COSINE,
            "maximum_decoder_coordinate_error_over_transition_path": MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH,
            "maximum_Q3_relative_drift": MAXIMUM_Q3_RELATIVE_DRIFT,
            "minimum_reconstruction_factor": MINIMUM_RECONSTRUCTION_FACTOR,
            "all_exact_rate_physical_gates_except_memoryless_hidden_fraction": True,
        },
        "growth_gates": {
            "maximum_full_collocation_defect": GROW_MAXIMUM_FULL_COLLOCATION_DEFECT,
            "maximum_normal_rate_defect": GROW_MAXIMUM_NORMAL_RATE_DEFECT,
            "minimum_rate_direction_cosine": GROW_MINIMUM_RATE_DIRECTION_COSINE,
            "maximum_decoder_coordinate_error_over_transition_path": GROW_MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH,
            "maximum_Q3_relative_drift": GROW_MAXIMUM_Q3_RELATIVE_DRIFT,
        },
        "hot_exit_event": {
            "hidden_secant_fraction_max": HIDDEN_SECANT_FRACTION_MAX,
            "consecutive_accepted_windows_required": HIDDEN_EXIT_PERSISTENCE_WINDOWS,
            "rank16_hidden_amplitude_min": RANK16_HIDDEN_AMPLITUDE_MIN,
            "macro_drift_from_seed_max": MAXIMUM_MACRO_DRIFT_FROM_SEED,
            "unchanged_from_legacy_exact_BDF_event_definition": True,
            "event_positive_window_holds_duration_for_confirmation": True,
            "event_requires_later_endpoint_refinement_and_held_out_replication": True,
        },
        "decision": {
            "event_observed": "authorize only a definitions-only hot-exit endpoint refinement manifest",
            "budget_exhausted_without_event": "authorize only a terminal phase-atlas prognosis",
            "window_failure": "stop; preserve the last accepted endpoint and diagnose the failed geometry",
            "complete_cycle_execution_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            name: helper._sha(path) for name, path in _decisive_inputs().items()
        },
        "frozen_source_hashes": {
            relative: helper._sha(ROOT / relative)
            for relative in (
                THIS_RUNNER,
                THIS_TEST,
                EXECUTION_RUNNER,
                EXECUTION_TEST,
                PHASE_SOURCE,
                FIXED_Q_SOURCE,
                EXACT_RATE_SOURCE,
                POST_RUNNER,
                LEGACY_EXIT_SOURCE,
            )
        },
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    rejected = architecture.rejected
    manifest_path = rejected.post.manifest.transition.manifest.cold.manifest.CANONICAL_MANIFEST
    summary_path = rejected.post.manifest.transition.manifest.cold.manifest.CANONICAL_SUMMARY
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": helper._sha(path), "scientific_status": "DEFINITIONS_ONLY"})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = helper._read(summary_path)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": PARENT_COMMIT, "latest_work_package": WORK_PACKAGE})
    helper._write_json(summary_path, catalog)


def _freeze() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("adaptive phase-atlas manifest already exists")
    parent = _validate_parent(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "adaptive_hot_exit_phase_atlas_contract.json", contract)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", parent)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "adaptive_phase_atlas_execution_authorized": True,
        "maximum_windows": MAXIMUM_WINDOWS,
        "maximum_truth_states_per_window": MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {"runner": THIS_RUNNER, "test": THIS_TEST, "implementation_commit": helper._git("rev-parse", "HEAD"), "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"), "python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Adaptive hot-exit phase-atlas manifest WP10c9d6c7c3b5c4f25ed", "", f"Classification: `{CLASSIFICATION}`.", "", "The selected event-driven architecture is extended prospectively by at most eight rank-adaptive, eight-node Lobatto windows. Each window uses no more than 15 unique exact continuous-rate states, no nonlinear fixed-Q roots, and no BDF microsteps.", "", "The historical hot-exit event definition is unchanged. A failed phase window is never propagated, and even an observed event authorizes only endpoint refinement and held-out replication—not a complete cycle or reduced slow evolution.", "")), encoding="utf-8")
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
