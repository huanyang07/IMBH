#!/usr/bin/env python3
"""Freeze the diagnosed fixed-Q one-root warm-policy certificate."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14i"
ARTIFACT = (
    "causal_inner_face36_fixed_q_warm_policy_manifest_"
    "wp10c9d6c7c3b5c4f24e14i"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
PARENT_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14h"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_warm_policy_manifest_"
    "wp10c9d6c7c3b5c4f24e14i.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_warm_policy_manifest_"
    "wp10c9d6c7c3b5c4f24e14i.py"
)
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_warm_policy_certificate_"
    "wp10c9d6c7c3b5c4f24e14j.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_face36_fixed_q_warm_policy_certificate_"
    "wp10c9d6c7c3b5c4f24e14j.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    EXECUTION_RUNNER,
    EXECUTION_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "tests/test_causal_inner_fixed_q.py",
)


CONTRACT = {
    "schema_version": 1,
    "work_package": WORK_PACKAGE,
    "definitions_only": True,
    "preserved_parent_classification": "bounded_continuation_failed",
    "warm_root": {
        "state": "primary_20ms",
        "timestep_seconds": 1.0e-7,
        "seed": "hash_locked_accepted_cold_1_checkpoint",
        "root_label": "warm_1",
        "carried_matrix_at_iteration_zero": True,
        "forced_initial_exact_assembly": False,
        "maximum_exact_assemblies": 1,
        "maximum_newton_iterations": 8,
        "maximum_line_search_iterations": 12,
        "refresh_policy": "on_line_search_failure_or_iteration_reserve",
        "primary_trigger_iteration": 6,
        "secondary_trigger_failed_relative_backtracks": 4,
        "maximum_scaled_residual": 1.0e-10,
        "all_existing_physical_and_history_gates": True,
    },
    "same_history_cold_control": {
        "required_only_after_warm_scientific_pass": True,
        "identical_start_state_history_target_scales_and_predictor": True,
        "fresh_exact_matrix_at_iteration_zero": True,
        "maximum_exact_assemblies": 2,
        "endpoint_scaled_state_tolerance": 1.0e-8,
        "reaction_action_relative_tolerance": 1.0e-8,
    },
    "cost_gate": {
        "warm_to_same_history_cold_wall_ratio_maximum": 0.75,
        "zero_exact_refresh_required": False,
        "residual_evaluation_ratio_is_reported": True,
        "wall_time_per_accepted_root_is_binding": True,
    },
    "classifications": [
        "warm_policy_scientific_and_cost_passed",
        "warm_policy_scientific_passed_cost_failed",
        "warm_policy_failed",
    ],
    "hard_stops": {
        "no_more_than_one_warm_root": True,
        "no_second_warm_continuation": True,
        "no_full_primary_retry": True,
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksums(directory: Path) -> dict[str, str]:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")
    return entries


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


def _parent_lock() -> dict:
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    if (
        summary["classification"]
        != "stale_carried_matrix_refresh_trigger_diagnosed"
        or not summary["passed"]
        or not summary["warm_policy_manifest_authorized"]
        or summary["warm_policy_execution_authorized"]
        or not metrics["positive_diagnosis"]
        or metrics["corrected_candidate_audit"]["maximum_scaled_residual"]
        > 1.0e-10
    ):
        raise RuntimeError("positive endpoint diagnosis changed")
    return {
        "summary": summary,
        "diagnostic_metrics": metrics,
        "decisive_hashes": {
            name: hashes[name]
            for name in (
                "summary.json",
                "metrics.json",
                "diagnostic_arrays.npz",
                "provenance.json",
            )
        },
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
        raise RuntimeError("warm-policy manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "warm_policy_manifest_frozen_one_root_and_same_history_control_"
            "authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "parent_classification_preserved": "bounded_continuation_failed",
        "one_root_warm_policy_execution_authorized": True,
        "full_primary_retry_authorized": False,
        "heldout_continuation_authorized": False,
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
            "parent_e14h_summary_sha256": parent["decisive_hashes"][
                "summary.json"
            ],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name) for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    files = ("execution_contract.json", "parent_lock.json", "provenance.json", "summary.json")
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in files)
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
