#!/usr/bin/env python3
"""Freeze one bounded post-transition phase-collocation window."""

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

import run_causal_inner_transition_phase_collocation_wp10c9d6c7c3b5c4f25e9 as transition  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ea"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25eb"
PARENT_COMMIT = "f8ca47bb08fdbdfb81e7c31acaa620a086bc4859"
PARENT_TREE = "cac601ebfc0d6d0e6b7aa779cc9217cf52514519"
CLASSIFICATION = "bounded_post_transition_rank4_lobatto_window_manifest_frozen"

NODE_COUNT = 8
FULL_DURATION_SECONDS = 2.0e-7
HALF_DURATION_SECONDS = FULL_DURATION_SECONDS / 2.0
RATE_BASIS_RANK = 4
PICARD_UPDATES = 1
MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS = 43

MAXIMUM_TRAINING_NORMAL_RATE_DEFECT = 1.0e-4
MAXIMUM_PROJECTED_COLLOCATION_DEFECT = 5.0e-2
MAXIMUM_FULL_COLLOCATION_DEFECT = 5.0e-2
MAXIMUM_NORMAL_RATE_DEFECT = 1.0e-2
MINIMUM_RATE_DIRECTION_COSINE = 0.995
MAXIMUM_MATCHED_ENDPOINT_COORDINATE_DEFECT = 5.0e-2
MAXIMUM_MATCHED_ENDPOINT_MACRO_DEFECT = 5.0e-2
MAXIMUM_MATCHED_ENDPOINT_STATE_DEFECT = 5.0e-2
MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH = 2.0e-3
MAXIMUM_Q3_RELATIVE_DRIFT = 5.0e-4
MINIMUM_RECONSTRUCTION_FACTOR = 1.0 - 1.0e-12

ARTIFACT = "causal_inner_post_transition_phase_window_manifest_wp10c9d6c7c3b5c4f25ea"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_post_transition_phase_window_manifest_wp10c9d6c7c3b5c4f25ea.py"
THIS_TEST = "tests/test_causal_inner_post_transition_phase_window_manifest_wp10c9d6c7c3b5c4f25ea.py"
EXECUTION_RUNNER = "scripts/run_causal_inner_post_transition_phase_window_wp10c9d6c7c3b5c4f25eb.py"
EXECUTION_TEST = "tests/test_causal_inner_post_transition_phase_window_wp10c9d6c7c3b5c4f25eb.py"
PHASE_SOURCE = "src/imri_qpe/layer3_minidisk_1d/phase_collocation.py"
FIXED_Q_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py"
EXACT_RATE_SOURCE = "scripts/run_causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy.py"
EXACT_CHART_SOURCE = "scripts/run_causal_inner_exact_geometric_470_chart_preflight_wp10c9d6c7c3b5c4f25de.py"
RATE_SOURCE = "scripts/run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_POST_TRANSITION_PHASE_WINDOW_MANIFEST_WP10C9D6C7C3B5C4F25EA_2026-08-21.md"
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return transition._helper()


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("post-transition manifest parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("post-transition manifest parent tree changed")
    hashes = helper._validate_checksums(transition.CANONICAL_DIRECTORY)
    summary = helper._read(transition.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["bounded_post_transition_manifest_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["hot_exit_observed"]
        or summary["predictive_cycle_authorized"]
    ):
        raise RuntimeError("transition phase-collocation authorization changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("post-transition manifest requires a clean tracked tree")
    return {"transition_result_hashes": hashes}


def _decisive_inputs() -> dict[str, Path]:
    terminal = transition.manifest.geometry.manifest._accepted_stage_directories()[-1]
    terminal_summary = _helper()._read(terminal / "summary.json")
    index = int(terminal_summary["step_index"])
    return {
        "transition_summary": transition.CANONICAL_DIRECTORY / "summary.json",
        "transition_metrics": transition.CANONICAL_DIRECTORY / "transition_collocation_metrics.json",
        "transition_witnesses": transition.CANONICAL_DIRECTORY / "transition_collocation_model_and_witnesses.npz",
        "transition_geometry": transition.manifest.manifest_geometry_path(),
        "terminal_checkpoint": terminal / f"checkpoint_step_{index:02d}.npz",
        "terminal_result": terminal / f"result_step_{index:02d}.npz",
    }


def _contract() -> dict:
    helper = _helper()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "initial_condition": {
            "state": "last accepted transition state at local time 1.1e-6 s",
            "fixed_Q_target": "losslessly carried accepted checkpoint target",
            "history_policy": "accepted history only; no rejected candidate enters the phase window",
        },
        "phase_discretization": {
            "method": "rank4 projected integral Legendre-Gauss-Lobatto collocation",
            "node_count": NODE_COUNT,
            "full_duration_seconds": FULL_DURATION_SECONDS,
            "matched_shadow": "two contiguous half-duration windows",
            "rate_basis": "rank4 right singular basis of eight normalized exact transition rates",
            "basis_rank": RATE_BASIS_RANK,
            "picard_updates": PICARD_UPDATES,
            "decoder": "terminal-anchored nonlinear displacement decoder",
            "accepted_endpoint_if_passed": "two-half-window endpoint",
            "no_sequential_BDF_microsteps": True,
        },
        "truth_budget": {
            "maximum_exact_continuous_fixed_Q_rate_calls": MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS,
            "new_nonlinear_fixed_Q_roots": 0,
            "new_BDF_microsteps": 0,
            "candidate_states_are_not_accepted_until_all_gates_pass": True,
        },
        "binding_gates": {
            "maximum_training_normal_rate_defect": MAXIMUM_TRAINING_NORMAL_RATE_DEFECT,
            "maximum_projected_collocation_defect": MAXIMUM_PROJECTED_COLLOCATION_DEFECT,
            "maximum_full_collocation_defect": MAXIMUM_FULL_COLLOCATION_DEFECT,
            "maximum_normal_rate_defect": MAXIMUM_NORMAL_RATE_DEFECT,
            "minimum_rate_direction_cosine": MINIMUM_RATE_DIRECTION_COSINE,
            "maximum_matched_endpoint_coordinate_defect": MAXIMUM_MATCHED_ENDPOINT_COORDINATE_DEFECT,
            "maximum_matched_endpoint_macro_defect": MAXIMUM_MATCHED_ENDPOINT_MACRO_DEFECT,
            "maximum_matched_endpoint_state_defect": MAXIMUM_MATCHED_ENDPOINT_STATE_DEFECT,
            "maximum_decoder_coordinate_error_over_transition_path": MAXIMUM_DECODER_COORDINATE_ERROR_OVER_TRANSITION_PATH,
            "maximum_Q3_relative_drift": MAXIMUM_Q3_RELATIVE_DRIFT,
            "minimum_reconstruction_factor": MINIMUM_RECONSTRUCTION_FACTOR,
            "all_exact_rate_physical_gates_except_memoryless_hidden_fraction": True,
        },
        "decision": {
            "pass_classification": "bounded_post_transition_rank4_phase_window_passed",
            "failure_classification": "bounded_post_transition_rank4_phase_window_rejected",
            "either_outcome_authorizes": "definitions_only_cycle_map_architecture_decision",
            "pass_does_not_claim_hot_exit": True,
            "predictive_cycle_execution_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            name: helper._sha(path) for name, path in _decisive_inputs().items()
        },
        "frozen_source_hashes": {
            name: helper._sha(ROOT / name)
            for name in (
                THIS_RUNNER,
                THIS_TEST,
                EXECUTION_RUNNER,
                EXECUTION_TEST,
                PHASE_SOURCE,
                FIXED_Q_SOURCE,
                EXACT_RATE_SOURCE,
                EXACT_CHART_SOURCE,
                RATE_SOURCE,
            )
        },
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = transition.manifest.cold.manifest.CANONICAL_MANIFEST
    summary_path = transition.manifest.cold.manifest.CANONICAL_SUMMARY
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
        raise RuntimeError("post-transition phase-window manifest already exists")
    parent = _validate_parent(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "post_transition_phase_window_contract.json", contract)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", parent)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "exact_continuous_rate_call_budget": MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS,
        "post_transition_phase_window_execution_authorized": True,
        "hot_exit_execution_authorized": False,
        "predictive_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {"runner": THIS_RUNNER, "test": THIS_TEST, "implementation_commit": helper._git("rev-parse", "HEAD"), "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"), "python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Bounded post-transition phase-window manifest WP10c9d6c7c3b5c4f25ea", "", f"Classification: `{CLASSIFICATION}`.", "", "One rank-4, eight-node Lobatto window of duration 0.2 microseconds and a matched two-half-window shadow are frozen prospectively. The budget is 43 exact continuous fixed-Q rate calls, zero nonlinear roots, and zero BDF microsteps.", "", "Either result leads only to a definitions-only cycle-map architecture decision. A pass does not claim a hot exit or authorize reduced slow evolution.", "")), encoding="utf-8")
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
