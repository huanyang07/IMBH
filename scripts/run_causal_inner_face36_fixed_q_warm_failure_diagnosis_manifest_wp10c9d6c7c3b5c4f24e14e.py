#!/usr/bin/env python3
"""Freeze the fixed-Q warm-failure diagnosis and repair manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14e"
ARTIFACT = (
    "causal_inner_face36_fixed_q_warm_failure_diagnosis_manifest_"
    "wp10c9d6c7c3b5c4f24e14e"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
E14D_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_warm_failure_diagnosis_"
    "manifest_wp10c9d6c7c3b5c4f24e14e.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_warm_failure_diagnosis_"
    "manifest_wp10c9d6c7c3b5c4f24e14e.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


CONTRACT = {
    "schema_version": 1,
    "work_package": WORK_PACKAGE,
    "definitions_only": True,
    "trajectory_may_execute_during_freeze": False,
    "preserved_parent_classification": "bounded_continuation_failed",
    "repair_preflight": {
        "physical_operator_may_change": False,
        "root_tolerance_may_change": False,
        "canonical_parent_evidence_may_be_rewritten": False,
        "required_solver_counters": [
            "total_broyden_updates",
            "broyden_updates_since_last_exact",
        ],
        "reset_updates_since_last_exact_on_every_exact_assembly": True,
        "legacy_solver_state_schema_must_load_explicitly": True,
        "legacy_counter_semantics_must_be_marked_untrusted": True,
        "failure_aware_canonical_fields": [
            "attempted_roots",
            "accepted_roots",
            "rejected_roots",
            "accepted_trajectory_horizon_seconds",
            "accepted_trajectory_cumulative_ledger",
            "rejected_candidate_diagnostic_ledgers",
            "planned_ladder_complete",
        ],
        "accepted_horizon_uses_only_accepted_roots": True,
        "accepted_ledgers_exclude_rejected_candidates": True,
        "profiling_requires_call_counts": True,
        "profiling_requires_exclusive_wall_times": True,
        "implementation_tests_must_run_without_a_nonlinear_root": True,
    },
    "endpoint_diagnostic_preflight": {
        "diagnostic_may_execute_during_this_package": False,
        "start_checkpoint": "checkpoint_cold_1.npz",
        "rejected_endpoint": "result_warm_1.npz",
        "committed_residual": 5.708109263036221e-9,
        "residual_reproduction": "bitwise",
        "maximum_exact_complete_jacobian_assemblies": 1,
        "maximum_exact_newton_corrections": 1,
        "prospective_line_search_alphas": [
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
        ],
        "continuation_state_may_be_constructed": False,
        "rejected_endpoint_may_enter_history": False,
        "binding_comparisons": [
            "exact_and_carried_correction_angle_in_equilibrated_coordinates",
            "exact_and_carried_correction_norm_ratio",
            "exact_jacobian_action_defect_on_carried_correction",
            "carried_matrix_action_defect_on_exact_correction",
            "exact_linear_solve_residual",
            "actual_post_correction_scaled_residual",
        ],
        "full_matrix_frobenius_defect_is_diagnostic_only": True,
        "all_existing_physical_storage_reaction_and_ledger_audits_required": (
            True
        ),
    },
    "diagnostic_classifications": {
        "positive": "stale_carried_matrix_refresh_trigger_diagnosed",
        "improved_not_passed": "endpoint_exact_diagnostic_inconclusive",
        "not_improved_or_solve_failed": "endpoint_exact_diagnostic_failed",
        "parent_rejection_may_be_reclassified": False,
    },
    "conditional_next_policy": {
        "authorized_only_after_positive_endpoint_diagnosis": True,
        "carried_matrix_at_iteration_zero": True,
        "forced_initial_exact_assembly": False,
        "maximum_exact_assemblies_per_warm_root": 1,
        "primary_refresh_trigger": (
            "beginning_of_iteration_maximum_iterations_minus_two_if_"
            "unconverged_and_no_current_root_exact_assembly"
        ),
        "secondary_refresh_trigger": (
            "no_merit_decrease_after_four_relative_backtracks"
        ),
        "unchanged_maximum_newton_iterations": 8,
        "unchanged_maximum_scaled_residual": 1.0e-10,
        "zero_refresh_is_not_a_cost_gate": True,
    },
    "execution_order": [
        "freeze_this_definitions_only_manifest",
        "repair_and_unit_certify_accounting_canonicalization_and_replay",
        "freeze_a_separate_endpoint_execution_manifest",
        "execute_only_one_nonpropagating_exact_endpoint_diagnostic",
        "stop_unless_the_positive_diagnostic_classification_is_obtained",
    ],
    "hard_stops": {
        "no_physical_root_in_this_manifest": True,
        "no_gate_relaxation": True,
        "no_iteration_budget_increase": True,
        "no_rejected_candidate_in_history": True,
        "no_full_four_root_retry": True,
        "no_heldout_continuation": True,
        "no_operational_timestep_search": True,
        "no_fixed_Q_micro_solver": True,
        "no_physical_microburst": True,
        "no_fast_averaging": True,
        "no_reduced_slow_evolution": True,
    },
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_tree_is_clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(
            ("git", "diff", "--cached", "--quiet"), cwd=ROOT
        ).returncode
        == 0
    )


def _validate_checksums(directory: Path) -> dict[str, str]:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")
    return entries


def _failure_lock() -> dict:
    hashes = _validate_checksums(E14D_DIRECTORY)
    summary = _read(E14D_DIRECTORY / "summary.json")
    metrics = _read(E14D_DIRECTORY / "metrics.json")
    if (
        summary["classification"] != "bounded_continuation_failed"
        or summary["scientific_passed"]
        or summary["accepted_main_BDF2_roots"] != 1
        or summary["attempted_main_BDF2_roots"] != 2
        or summary["accepted_main_horizon_seconds"] != 1.0e-7
        or metrics["failure_stage"] != "binding root warm_1 failed"
        or metrics["main_roots"]["cold_1"]["accepted"] is not True
        or metrics["main_roots"]["warm_1"]["accepted"] is not False
        or metrics["main_roots"]["warm_1"]["maximum_scaled_residual"]
        != CONTRACT["endpoint_diagnostic_preflight"]["committed_residual"]
    ):
        raise RuntimeError("e14d binding rejection changed")
    inventories = {}
    required_arrays = {
        "checkpoint_cold_1.npz": {
            "current_primitive_charts",
            "previous_primitive_charts",
            "previous_primitive_increment",
            "previous_mapped_storage_increment",
            "previous_responsive_height_storage_increment",
            "q3_target",
            "constraint_row_scales",
            "solver_bordered_matrix_raw",
            "solver_anchor_primitive_charts",
        },
        "result_warm_1.npz": {
            "primitive_charts",
            "primitive_increment",
            "multipliers",
            "augmented_scaled_residual",
            "raw_solver_matrix",
            "metrics_json",
        },
    }
    for name, required in required_arrays.items():
        with np.load(E14D_DIRECTORY / name, allow_pickle=False) as source:
            if not required.issubset(source.files):
                raise RuntimeError(f"e14d diagnostic inventory changed: {name}")
            inventories[name] = sorted(source.files)
    return {
        "schema_version": 1,
        "artifact": E14D_DIRECTORY.name,
        "summary": summary,
        "decisive_hashes": {
            name: hashes[name]
            for name in (
                "summary.json",
                "metrics.json",
                "checkpoint_cold_1.npz",
                "result_cold_1.npz",
                "result_warm_1.npz",
                "decisive_arrays.npz",
                "execution_identity.json",
                "provenance.json",
            )
        },
        "array_inventories": inventories,
    }


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    failure = _failure_lock()
    if not _tracked_tree_is_clean():
        raise RuntimeError("warm-failure manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "warm_failure_diagnosis_manifest_frozen_"
            "accounting_repair_preflight_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "accounting_repair_preflight_authorized": True,
        "endpoint_diagnostic_execution_authorized": False,
        "warm_policy_execution_authorized": False,
        "full_primary_retry_authorized": False,
        "heldout_continuation_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next_action": (
            "implement_and_unit_certify_failure_aware_accounting_"
            "solver_counters_and_endpoint_replay_only"
        ),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(ARTIFACT_DIRECTORY / "diagnosis_contract.json", CONTRACT)
    _write(ARTIFACT_DIRECTORY / "parent_failure_lock.json", failure)
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in SOURCE_FILES
            },
            "parent_e14d_summary_sha256": failure["decisive_hashes"][
                "summary.json"
            ],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment_at_freeze": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT
            },
        },
    )
    files = (
        "diagnosis_contract.json",
        "parent_failure_lock.json",
        "provenance.json",
        "summary.json",
    )
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in files
        ),
        encoding="utf-8",
    )
    _catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        raise SystemExit("select --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
