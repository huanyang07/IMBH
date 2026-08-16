#!/usr/bin/env python3
"""Freeze one nonpropagating exact-Jacobian endpoint diagnostic."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14g"
ARTIFACT = (
    "causal_inner_face36_fixed_q_exact_endpoint_diagnostic_manifest_"
    "wp10c9d6c7c3b5c4f24e14g"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
PARENT_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_warm_failure_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14f"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "manifest_wp10c9d6c7c3b5c4f24e14g.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "manifest_wp10c9d6c7c3b5c4f24e14g.py"
)
DIAGNOSTIC_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14h.py"
)
DIAGNOSTIC_TEST = (
    "tests/test_causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14h.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    DIAGNOSTIC_RUNNER,
    DIAGNOSTIC_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "scripts/run_causal_inner_face36_fixed_q_warm_failure_implementation_"
    "preflight_wp10c9d6c7c3b5c4f24e14f.py",
)


CONTRACT = {
    "schema_version": 1,
    "work_package": WORK_PACKAGE,
    "definitions_only": True,
    "parent_classification_preserved": "bounded_continuation_failed",
    "authorized_diagnostic": {
        "state": "primary_20ms",
        "timestep_seconds": 1.0e-7,
        "start_checkpoint": "checkpoint_cold_1.npz",
        "rejected_endpoint": "result_warm_1.npz",
        "committed_maximum_scaled_residual": 5.708109263036221e-9,
        "residual_reproduction": "bitwise",
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
        "unchanged_maximum_scaled_primitive_change": 5.0e-3,
        "all_existing_physical_storage_reaction_and_ledger_gates": True,
        "continuation_state_may_be_constructed": False,
        "rejected_or_corrected_candidate_may_enter_history": False,
    },
    "binding_diagnostics": [
        "exact_and_carried_correction_angle",
        "exact_and_carried_correction_norm_ratio",
        "exact_jacobian_action_defect_on_carried_correction",
        "carried_matrix_action_defect_on_exact_correction",
        "exact_linear_solve_residual",
        "actual_post_correction_scaled_residual",
    ],
    "diagnostic_only": ["full_matrix_relative_frobenius_defect"],
    "classifications": {
        "positive": "stale_carried_matrix_refresh_trigger_diagnosed",
        "improved_not_passed": "endpoint_exact_diagnostic_inconclusive",
        "failed": "endpoint_exact_diagnostic_failed",
    },
    "hard_stops": {
        "no_parent_reclassification": True,
        "no_continuation_state": True,
        "no_trajectory_advance": True,
        "no_warm_policy_execution": True,
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
    replay = _read(PARENT_DIRECTORY / "endpoint_replay.json")
    if (
        not summary["passed"]
        or not summary["accounting_repair_certified"]
        or not summary["endpoint_diagnostic_manifest_authorized"]
        or summary["endpoint_diagnostic_execution_authorized"]
        or not replay["bitwise_residual_reproduction"]
        or replay["replayed_maximum_scaled_residual"]
        != CONTRACT["authorized_diagnostic"][
            "committed_maximum_scaled_residual"
        ]
    ):
        raise RuntimeError("e14f endpoint-manifest authorization changed")
    return {
        "summary": summary,
        "endpoint_replay": replay,
        "decisive_hashes": {
            name: hashes[name]
            for name in (
                "summary.json",
                "endpoint_replay.json",
                "endpoint_replay_arrays.npz",
                "failure_aware_accounting.json",
                "legacy_counter_reconstruction.json",
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
        raise RuntimeError("endpoint diagnostic manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "exact_endpoint_diagnostic_manifest_frozen_"
            "one_nonpropagating_correction_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "parent_classification_preserved": "bounded_continuation_failed",
        "exact_endpoint_diagnostic_execution_authorized": True,
        "warm_policy_execution_authorized": False,
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
            "parent_e14f_summary_sha256": parent["decisive_hashes"][
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
    files = ("execution_contract.json", "parent_lock.json", "provenance.json", "summary.json")
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
