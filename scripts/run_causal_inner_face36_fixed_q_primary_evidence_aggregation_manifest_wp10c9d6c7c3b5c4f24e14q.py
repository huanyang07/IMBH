#!/usr/bin/env python3
"""Freeze aggregation of the already executed primary continuation evidence."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14q"
ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_evidence_aggregation_manifest_"
    "wp10c9d6c7c3b5c4f24e14q"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
RETRY_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_retry_wp10c9d6c7c3b5c4f24e14l"
)
DIAGNOSIS_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_cold_shadow_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14n"
)
POLICY_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_same_history_equivalence_policy_"
    "wp10c9d6c7c3b5c4f24e14p"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_primary_evidence_aggregation_"
    "manifest_wp10c9d6c7c3b5c4f24e14q.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_primary_evidence_aggregation_"
    "manifest_wp10c9d6c7c3b5c4f24e14q.py"
)
AGGREGATOR = (
    "scripts/run_causal_inner_face36_fixed_q_primary_evidence_aggregation_"
    "wp10c9d6c7c3b5c4f24e14r.py"
)
AGGREGATOR_TEST = (
    "tests/test_causal_inner_face36_fixed_q_primary_evidence_aggregation_"
    "wp10c9d6c7c3b5c4f24e14r.py"
)
SOURCE_FILES = (THIS_RUNNER, THIS_TEST, AGGREGATOR, AGGREGATOR_TEST)


CONTRACT = {
    "schema_version": 1,
    "work_package": WORK_PACKAGE,
    "definitions_only": True,
    "historical_parent_classification_preserved": "bounded_continuation_failed",
    "aggregation_only": True,
    "physical_root_execution_authorized": False,
    "binding_primary_evidence": {
        "accepted_main_BDF2_roots": 4,
        "accepted_main_horizon_seconds": 4.0e-7,
        "all_main_root_acceptance_gates": True,
        "suffix_replay_bitwise": True,
        "cumulative_ledger_passed": True,
        "matched_endpoint_half_step_audit_passed": True,
        "same_history_control_accepted": True,
        "same_history_cost_passed": True,
        "maximum_warm_to_cold_wall_time_ratio": 0.75,
        "maximum_warm_to_cold_residual_evaluation_ratio": 0.75,
        "certified_control_comparison_residual": 1.0e-12,
        "maximum_scaled_state_difference": 1.0e-8,
        "maximum_reaction_action_relative_difference": 1.0e-8,
        "all_polished_control_audits": True,
    },
    "decision": {
        "pass": "primary_bounded_continuation_evidence_certified",
        "fail": "primary_evidence_aggregation_failed",
        "pass_authorizes_only": "definitions_only_heldout_continuation_manifest",
    },
    "hard_stops": {
        "no_historical_parent_reclassification": True,
        "no_new_physical_root": True,
        "no_trajectory_advance": True,
        "no_primary_retry": True,
        "no_heldout_execution": True,
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
    retry_hashes = _checksums(RETRY_DIRECTORY)
    diagnosis_hashes = _checksums(DIAGNOSIS_DIRECTORY)
    policy_hashes = _checksums(POLICY_DIRECTORY)
    retry_summary = _read(RETRY_DIRECTORY / "summary.json")
    retry_metrics = _read(RETRY_DIRECTORY / "metrics.json")
    diagnosis_summary = _read(DIAGNOSIS_DIRECTORY / "summary.json")
    policy_summary = _read(POLICY_DIRECTORY / "summary.json")
    policy_metrics = _read(POLICY_DIRECTORY / "metrics.json")
    if (
        retry_summary["classification"] != "bounded_continuation_failed"
        or retry_summary["accepted_main_BDF2_roots"] != 4
        or retry_summary["accepted_main_horizon_seconds"] != 4.0e-7
        or not retry_summary["planned_ladder_complete"]
        or diagnosis_summary["classification"]
        != "cold_shadow_residual_limited_action_equivalence_diagnosed"
        or not diagnosis_summary["passed"]
        or policy_summary["classification"]
        != "same_history_equivalence_policy_certified"
        or not policy_summary["passed"]
        or not policy_summary["primary_evidence_aggregation_manifest_authorized"]
        or not policy_metrics["policy_passed"]
        or retry_metrics["failure_stage"] != "same_history_cold_shadow"
    ):
        raise RuntimeError("primary aggregation authorization changed")
    return {
        "historical_retry_summary": retry_summary,
        "historical_retry_metrics": retry_metrics,
        "positive_diagnosis_summary": diagnosis_summary,
        "policy_certificate_summary": policy_summary,
        "policy_certificate_metrics": policy_metrics,
        "package_hashes": {
            "historical_retry": retry_hashes,
            "positive_diagnosis": diagnosis_hashes,
            "policy_certificate": policy_hashes,
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
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
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
        raise RuntimeError("primary aggregation manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "primary_evidence_aggregation_manifest_frozen_evaluation_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "historical_parent_classification_preserved": "bounded_continuation_failed",
        "primary_evidence_aggregation_authorized": True,
        "heldout_continuation_manifest_authorized": False,
        "heldout_continuation_execution_authorized": False,
        "operational_timestep_study_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(ARTIFACT_DIRECTORY / "aggregation_contract.json", CONTRACT)
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
    files = ("aggregation_contract.json", "parent_lock.json", "provenance.json", "summary.json")
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
