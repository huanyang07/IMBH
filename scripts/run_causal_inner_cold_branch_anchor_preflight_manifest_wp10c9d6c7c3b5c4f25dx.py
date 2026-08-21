#!/usr/bin/env python3
"""Freeze fail-fast exact fixed-Q preflight of saved cold candidates."""

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

import run_causal_inner_transition_terminal_prognosis_wp10c9d6c7c3b5c4f25dw as decision  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dx"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dy"
PARENT_COMMIT = "9f2fa0b6bd49ec2a3cb6ffdbd2a7fa090c8a5c0c"
PARENT_TREE = "5e93ed073ab81dadbeed5e48d827bc1f57efa470"
CLASSIFICATION = "saved_cold_candidate_exact_fixed_Q_anchor_preflight_manifest_frozen"

CANDIDATE_TIMES_SECONDS = (0.012, 0.008, 0.005, 0.002)
HIDDEN_FRACTION_GATE = 0.25
FIXED_Q_TANGENCY_GATE = 1.0e-12
DECOMPOSITION_GATE = 1.0e-10
COORDINATE_JACOBIAN_CONDITION_GATE = 1.0e8
REACTION_LEDGER_GATE = 1.0e-12
SCHUR_CONDITION_GATE = 1.0e8
RECONSTRUCTION_GATE = 1.0 - 1.0e-12
HEIGHT_RATIO_GATE = 0.5
OPTICAL_DEPTH_GATE = 1.0
MAXIMUM_EXACT_RATE_EVALUATIONS = len(CANDIDATE_TIMES_SECONDS)

ARTIFACT = "causal_inner_cold_branch_anchor_preflight_manifest_wp10c9d6c7c3b5c4f25dx"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_cold_branch_anchor_preflight_manifest_wp10c9d6c7c3b5c4f25dx.py"
THIS_TEST = "tests/test_causal_inner_cold_branch_anchor_preflight_manifest_wp10c9d6c7c3b5c4f25dx.py"
EXECUTION_RUNNER = "scripts/run_causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy.py"
EXECUTION_TEST = "tests/test_causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COLD_BRANCH_ANCHOR_PREFLIGHT_"
    "MANIFEST_WP10C9D6C7C3B5C4F25DX_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CANDIDATE_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_hybrid_candidate_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25dc"
)
SCREEN_DIRECTORY = decision.manifest.COLD_SCREEN_DIRECTORY
TANGENT_ARRAYS = decision.manifest.tube.manifest.geometry_manifest.TANGENT_ARRAYS


def _validate_parent(*, require_clean: bool) -> dict:
    helper = decision.manifest.tube.manifest.geometry
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("cold-anchor parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("cold-anchor parent tree changed")
    decision_hashes = helper._validate_checksums(decision.CANONICAL_DIRECTORY)
    candidate_hashes = helper._validate_checksums(CANDIDATE_DIRECTORY)
    screen_hashes = helper._validate_checksums(SCREEN_DIRECTORY)
    summary = helper._read(decision.CANONICAL_DIRECTORY / "summary.json")
    screen = helper._read(SCREEN_DIRECTORY / "branch_candidate_screen_metrics.json")
    if (
        not summary["passed"]
        or not summary["branch_pseudo_arclength_architecture_selected"]
        or not summary["cold_branch_root_manifest_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or screen["selected_cold_candidate"] != "full_model_12ms"
        or screen["selected_hot_candidate"] is not None
    ):
        raise RuntimeError("cold-anchor authorization changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cold-anchor manifest requires a clean tracked tree")
    return {
        "decision_hashes": decision_hashes,
        "candidate_hashes": candidate_hashes,
        "screen_hashes": screen_hashes,
    }


def _contract() -> dict:
    helper = decision.manifest.tube.manifest.geometry
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "candidate_policy": {
            "times_seconds_in_fail_fast_order": CANDIDATE_TIMES_SECONDS,
            "evaluate_one_candidate_at_a_time": True,
            "stop_at_first_complete_pass": True,
            "all_states_previously_revealed_and_accepted": True,
            "sealed_16ms_state_forbidden": True,
            "20ms_transition_state_forbidden": True,
        },
        "binding_gates": {
            "hidden_coordinate_rate_fraction_max": HIDDEN_FRACTION_GATE,
            "fixed_Q_tangency_max": FIXED_Q_TANGENCY_GATE,
            "coordinate_decomposition_max": DECOMPOSITION_GATE,
            "coordinate_jacobian_rank_equal": 470,
            "coordinate_jacobian_condition_max": COORDINATE_JACOBIAN_CONDITION_GATE,
            "reaction_ledger_max": REACTION_LEDGER_GATE,
            "Schur_rank_equal": 3,
            "Schur_condition_max": SCHUR_CONDITION_GATE,
            "minimum_reconstruction_factor": RECONSTRUCTION_GATE,
            "maximum_height_ratio": HEIGHT_RATIO_GATE,
            "minimum_optical_depth": OPTICAL_DEPTH_GATE,
        },
        "budgets": {
            "maximum_exact_fixed_Q_rate_evaluations": MAXIMUM_EXACT_RATE_EVALUATIONS,
            "complete_generator_assemblies": 0,
            "hidden_branch_roots": 0,
            "propagated_states": 0,
            "new_transition_microsteps": 0,
        },
        "decision_policy": {
            "first_complete_pass": "authorize_definitions_only_single_cold_hidden_root",
            "no_candidate_passes": "cold_branch_seed_not_established_stop_branch_root",
            "root_not_executed_in_preflight": True,
            "hot_branch_and_complete_impulse_blocked": True,
            "reduced_cycle_blocked": True,
        },
        "input_hashes": {
            "decision_summary": helper._sha(decision.CANONICAL_DIRECTORY / "summary.json"),
            "candidate_arrays": helper._sha(CANDIDATE_DIRECTORY / "candidate_geometry_arrays.npz"),
            "screen_arrays": helper._sha(SCREEN_DIRECTORY / "branch_candidate_screen_arrays.npz"),
            "tangent_arrays": helper._sha(TANGENT_ARRAYS),
        },
        "frozen_source_hashes": {
            THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
            THIS_TEST: helper._sha(ROOT / THIS_TEST),
            EXECUTION_RUNNER: helper._sha(ROOT / EXECUTION_RUNNER),
            EXECUTION_TEST: helper._sha(ROOT / EXECUTION_TEST),
        },
    }


def _update_catalog(summary: dict) -> None:
    helper = decision.manifest.tube.manifest.geometry
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
                    "sha256": helper._sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
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
    helper._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    helper = decision.manifest.tube.manifest.geometry
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("cold-anchor preflight manifest already exists")
    locks = _validate_parent(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "cold_anchor_contract.json", contract)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {"parent_commit": PARENT_COMMIT, "parent_tree": PARENT_TREE, **locks},
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "candidate_count": len(CANDIDATE_TIMES_SECONDS),
        "new_truth_calls": 0,
        "branch_root_executed": False,
        "hot_branch_blocked": True,
        "reduced_cycle_blocked": True,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "runner": THIS_RUNNER,
            "tests": [THIS_TEST, EXECUTION_TEST],
            "execution_runner": EXECUTION_RUNNER,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": contract["frozen_source_hashes"],
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Cold-branch anchor preflight manifest WP10c9d6c7c3b5c4f25dx",
                "",
                "Previously revealed accepted states are tested at 12, 8, 5, and 2 ms in fail-fast order. Each call evaluates the exact fixed-Q rate and exact coordinate Jacobian; the first complete hidden-fraction/physical pass stops the ladder.",
                "",
                "No generator, hidden root, propagation, transition step, sealed 16 ms call, hot branch, or reduced cycle is authorized here.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("use --freeze")
    print(json.dumps(decision.manifest.tube.manifest.geometry._plain(_freeze()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
