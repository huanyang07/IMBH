#!/usr/bin/env python3
"""Freeze one nonpropagating cold-shadow exact-endpoint diagnostic."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14m"
ARTIFACT = (
    "causal_inner_face36_fixed_q_cold_shadow_endpoint_diagnostic_manifest_"
    "wp10c9d6c7c3b5c4f24e14m"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
PARENT_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_cold_shadow_endpoint_"
    "diagnostic_manifest_wp10c9d6c7c3b5c4f24e14m.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_cold_shadow_endpoint_"
    "diagnostic_manifest_wp10c9d6c7c3b5c4f24e14m.py"
)
DIAGNOSTIC_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_cold_shadow_endpoint_"
    "diagnostic_wp10c9d6c7c3b5c4f24e14n.py"
)
DIAGNOSTIC_TEST = (
    "tests/test_causal_inner_face36_fixed_q_cold_shadow_endpoint_"
    "diagnostic_wp10c9d6c7c3b5c4f24e14n.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    DIAGNOSTIC_RUNNER,
    DIAGNOSTIC_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "scripts/run_causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14h.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py",
)


CONTRACT = {
    "schema_version": 1,
    "work_package": WORK_PACKAGE,
    "definitions_only": True,
    "parent_classification_preserved": "bounded_continuation_failed",
    "authorized_diagnostic": {
        "state": "primary_20ms",
        "timestep_seconds": 1.0e-7,
        "common_start_checkpoint": "checkpoint_warm_1.npz",
        "warm_endpoint": "result_warm_2.npz",
        "cold_shadow_endpoint": "result_cold_shadow.npz",
        "committed_warm_maximum_scaled_residual": 5.048217216618925e-13,
        "committed_cold_maximum_scaled_residual": 6.398284679853816e-11,
        "committed_scaled_state_absolute_defect": 7.859135564558528e-11,
        "committed_reaction_action_relative_defect": 2.8666087608919947e-8,
        "saved_residual_and_reaction_action_reproduction": "bitwise",
        "maximum_exact_complete_jacobian_assemblies": 1,
        "maximum_exact_newton_corrections": 1,
        "relative_line_search_factors": [
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
        ],
        "unchanged_maximum_scaled_residual": 1.0e-10,
        "unchanged_maximum_scaled_state_difference": 1.0e-8,
        "unchanged_maximum_reaction_action_relative_difference": 1.0e-8,
        "all_existing_physical_storage_reaction_and_ledger_gates": True,
        "continuation_state_may_be_constructed": False,
        "candidate_may_enter_history": False,
    },
    "binding_diagnostics": [
        "bitwise_saved_warm_and_cold_residual_reproduction",
        "bitwise_saved_warm_and_cold_reaction_action_reproduction",
        "exact_linear_solve_relative_residual",
        "corrected_cold_complete_residual",
        "corrected_cold_physical_acceptance",
        "corrected_cold_to_warm_scaled_state_absolute_defect",
        "corrected_cold_to_warm_reaction_action_relative_defect",
    ],
    "diagnostic_only": [
        "cold_action_change_per_scaled_state_correction",
        "reaction_action_defect_reduction_factor",
        "Schur_condition_number",
    ],
    "classifications": {
        "positive": "cold_shadow_residual_limited_action_equivalence_diagnosed",
        "improved_not_passed": "cold_shadow_exact_endpoint_diagnostic_inconclusive",
        "failed": "cold_shadow_exact_endpoint_diagnostic_failed",
    },
    "hard_stops": {
        "no_parent_reclassification": True,
        "no_continuation_state": True,
        "no_trajectory_advance": True,
        "no_primary_retry": True,
        "no_heldout_continuation": True,
        "no_operational_timestep_search": True,
        "no_fixed_Q_micro_solver": True,
        "no_physical_microburst": True,
        "no_fast_averaging": True,
        "no_reduced_slow_evolution": True,
        "no_gate_relaxation": True,
    },
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_tree_is_clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(
            ("git", "diff", "--cached", "--quiet"), cwd=ROOT
        ).returncode
        == 0
    )


def _checksums(directory: Path) -> dict[str, str]:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")
    return entries


def _parent_lock() -> dict:
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    shadow = metrics["same_history_cold_shadow"]
    half = metrics["matched_endpoint_half_step_audit"]
    diagnostic = CONTRACT["authorized_diagnostic"]
    if (
        summary["classification"] != "bounded_continuation_failed"
        or summary["scientific_passed"]
        or not summary["trajectory_executed"]
        or summary["accepted_main_BDF2_roots"] != 4
        or metrics["failure_stage"] != "same_history_cold_shadow"
        or not metrics["replay"]["passed"]
        or not metrics["cumulative_ledger_passed"]
        or not half["passed"]
        or not shadow["executed"]
        or shadow["scientific_passed"]
        or not shadow["cost_passed"]
        or not shadow["root"]["accepted"]
        or shadow["root"]["maximum_scaled_residual"]
        != diagnostic["committed_cold_maximum_scaled_residual"]
        or metrics["main_roots"]["warm_2"]["maximum_scaled_residual"]
        != diagnostic["committed_warm_maximum_scaled_residual"]
        or shadow["scaled_state_absolute_defect"]
        != diagnostic["committed_scaled_state_absolute_defect"]
        or shadow["reaction_action_relative_defect"]
        != diagnostic["committed_reaction_action_relative_defect"]
    ):
        raise RuntimeError("e14l cold-shadow diagnostic authorization changed")
    decisive_names = (
        "summary.json",
        "metrics.json",
        "checkpoint_warm_1.npz",
        "result_warm_2.npz",
        "result_cold_shadow.npz",
        "provenance.json",
    )
    return {
        "summary": summary,
        "decisive_metrics": {
            "failure_stage": metrics["failure_stage"],
            "replay_passed": metrics["replay"]["passed"],
            "cumulative_ledger_passed": metrics["cumulative_ledger_passed"],
            "matched_half_step_passed": half["passed"],
            "same_history_cold_shadow": shadow,
        },
        "decisive_hashes": {name: hashes[name] for name in decisive_names},
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
    parent = _parent_lock()
    if not _tracked_tree_is_clean():
        raise RuntimeError("cold-shadow diagnostic manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "cold_shadow_endpoint_diagnostic_manifest_frozen_"
            "one_nonpropagating_exact_correction_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "parent_classification_preserved": "bounded_continuation_failed",
        "cold_shadow_endpoint_diagnostic_execution_authorized": True,
        "primary_retry_authorized": False,
        "heldout_continuation_authorized": False,
        "operational_timestep_study_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(ARTIFACT_DIRECTORY / "execution_contract.json", CONTRACT)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", parent)
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
            "parent_e14l_summary_sha256": parent["decisive_hashes"][
                "summary.json"
            ],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
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
    files = (
        "execution_contract.json",
        "parent_lock.json",
        "provenance.json",
        "summary.json",
    )
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in files),
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
