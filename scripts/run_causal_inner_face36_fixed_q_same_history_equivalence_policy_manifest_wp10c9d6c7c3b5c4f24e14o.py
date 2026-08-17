#!/usr/bin/env python3
"""Freeze the control-only same-history endpoint-equivalence policy."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14o"
ARTIFACT = (
    "causal_inner_face36_fixed_q_same_history_equivalence_policy_manifest_"
    "wp10c9d6c7c3b5c4f24e14o"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
DIAGNOSIS_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_cold_shadow_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14n"
)
RETRY_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_same_history_equivalence_"
    "policy_manifest_wp10c9d6c7c3b5c4f24e14o.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_same_history_equivalence_"
    "policy_manifest_wp10c9d6c7c3b5c4f24e14o.py"
)
CERTIFICATE_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_same_history_equivalence_"
    "policy_wp10c9d6c7c3b5c4f24e14p.py"
)
CERTIFICATE_TEST = (
    "tests/test_causal_inner_face36_fixed_q_same_history_equivalence_"
    "policy_wp10c9d6c7c3b5c4f24e14p.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    CERTIFICATE_RUNNER,
    CERTIFICATE_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "scripts/run_causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14h.py",
    "scripts/run_causal_inner_face36_fixed_q_cold_shadow_endpoint_"
    "diagnostic_wp10c9d6c7c3b5c4f24e14n.py",
)


CONTRACT = {
    "schema_version": 1,
    "work_package": WORK_PACKAGE,
    "definitions_only": True,
    "historical_parent_classification_preserved": "bounded_continuation_failed",
    "policy_scope": "nonpropagating_same_history_equivalence_controls_only",
    "production_step_acceptance": {
        "maximum_scaled_residual": 1.0e-10,
        "unchanged": True,
    },
    "equivalence_control_policy": {
        "accepted_control_root_required_before_polish": True,
        "maximum_scaled_residual_before_state_action_comparison": 1.0e-12,
        "maximum_scaled_state_difference": 1.0e-8,
        "maximum_reaction_action_relative_difference": 1.0e-8,
        "endpoint_polish_trigger": (
            "accepted_control_residual_above_1e-12_or_"
            "state_action_equivalence_not_yet_closed"
        ),
        "maximum_endpoint_polish_exact_assemblies": 1,
        "maximum_endpoint_polish_corrections": 1,
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
        "all_existing_physical_storage_reaction_and_ledger_gates": True,
        "polished_control_may_define_history": False,
        "polished_control_may_define_trajectory": False,
        "multiplier_coordinate_equality_binding": False,
        "physical_reaction_action_equality_binding": True,
    },
    "authorized_certificate": {
        "common_start_checkpoint": "checkpoint_warm_1.npz",
        "warm_reference": "result_warm_2.npz",
        "accepted_control": "result_cold_shadow.npz",
        "maximum_exact_assemblies": 1,
        "maximum_exact_corrections": 1,
        "accepted_trajectory_horizon_seconds_added": 0.0,
    },
    "classifications": {
        "pass": "same_history_equivalence_policy_certified",
        "fail": "same_history_equivalence_policy_failed",
    },
    "hard_stops": {
        "no_historical_parent_reclassification": True,
        "no_production_root_gate_change": True,
        "no_continuation_state": True,
        "no_trajectory_advance": True,
        "no_primary_retry": True,
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
    diagnosis_hashes = _checksums(DIAGNOSIS_DIRECTORY)
    retry_hashes = _checksums(RETRY_DIRECTORY)
    diagnosis_summary = _read(DIAGNOSIS_DIRECTORY / "summary.json")
    diagnosis_metrics = _read(DIAGNOSIS_DIRECTORY / "metrics.json")
    retry_summary = _read(RETRY_DIRECTORY / "summary.json")
    if (
        diagnosis_summary["classification"]
        != "cold_shadow_residual_limited_action_equivalence_diagnosed"
        or not diagnosis_summary["passed"]
        or not diagnosis_summary["same_history_equivalence_policy_manifest_authorized"]
        or diagnosis_summary["same_history_equivalence_policy_execution_authorized"]
        or not diagnosis_metrics["positive_diagnosis"]
        or diagnosis_metrics["corrected_candidate_audit"]["maximum_scaled_residual"]
        > 1.0e-12
        or diagnosis_metrics["corrected_to_warm_comparison"][
            "scaled_state_absolute_defect"
        ]
        > 1.0e-8
        or diagnosis_metrics["corrected_to_warm_comparison"][
            "reaction_action_relative_defect"
        ]
        > 1.0e-8
        or retry_summary["classification"] != "bounded_continuation_failed"
        or retry_summary["accepted_main_BDF2_roots"] != 4
    ):
        raise RuntimeError("e14n equivalence-policy authorization changed")
    return {
        "diagnosis_summary": diagnosis_summary,
        "diagnosis_metrics": diagnosis_metrics,
        "historical_retry_summary": retry_summary,
        "decisive_hashes": {
            "diagnosis_summary.json": diagnosis_hashes["summary.json"],
            "diagnosis_metrics.json": diagnosis_hashes["metrics.json"],
            "diagnosis_arrays.npz": diagnosis_hashes["diagnostic_arrays.npz"],
            "retry_summary.json": retry_hashes["summary.json"],
            "retry_checkpoint_warm_1.npz": retry_hashes["checkpoint_warm_1.npz"],
            "retry_result_warm_2.npz": retry_hashes["result_warm_2.npz"],
            "retry_result_cold_shadow.npz": retry_hashes[
                "result_cold_shadow.npz"
            ],
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
        raise RuntimeError("same-history policy manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "same_history_equivalence_policy_manifest_frozen_"
            "nonpropagating_certificate_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "historical_parent_classification_preserved": "bounded_continuation_failed",
        "same_history_equivalence_policy_certificate_authorized": True,
        "primary_evidence_aggregation_manifest_authorized": False,
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
            "diagnosis_summary_sha256": parent["decisive_hashes"][
                "diagnosis_summary.json"
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
