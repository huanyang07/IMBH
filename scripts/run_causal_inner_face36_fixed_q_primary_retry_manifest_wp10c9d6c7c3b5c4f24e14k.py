#!/usr/bin/env python3
"""Freeze the iteration-reserve primary fixed-Q continuation retry."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

e14c = importlib.import_module(
    "run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "manifest_wp10c9d6c7c3b5c4f24e14c"
)

WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14k"
ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_retry_manifest_"
    "wp10c9d6c7c3b5c4f24e14k"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
PARENT_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_warm_policy_certificate_"
    "wp10c9d6c7c3b5c4f24e14j"
)
SEED_DIRECTORY = e14c.E14B_DIRECTORY
CANONICAL_SEED = e14c.CANONICAL_SEED
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_manifest_"
    "wp10c9d6c7c3b5c4f24e14k.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_primary_retry_manifest_"
    "wp10c9d6c7c3b5c4f24e14k.py"
)
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    EXECUTION_RUNNER,
    EXECUTION_TEST,
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
    "scripts/run_causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1.py",
)


CONTRACT = copy.deepcopy(e14c.CONTRACT)
CONTRACT.update(
    {
        "work_package": WORK_PACKAGE,
        "parent_work_packages": [
            "WP10c9d6c7c3b5c4f24e14b",
            "WP10c9d6c7c3b5c4f24e14j",
        ],
        "supersedes_execution_contract": "WP10c9d6c7c3b5c4f24e14c",
        "preserved_parent_classification": "bounded_continuation_failed",
    }
)
CONTRACT["solver_contract"].update(
    {
        "warm_refresh_policy": (
            "on_line_search_failure_or_iteration_reserve"
        ),
        "warm_iteration_reserve_trigger": 6,
        "warm_failed_relative_backtrack_trigger": 4,
        "maximum_exact_assemblies_per_warm_root": 1,
        "zero_refresh_warm_root_required": False,
    }
)
CONTRACT["same_history_cold_shadow"].update(
    {
        "cost_metric": "warm_to_same_history_cold_wall_time_ratio",
        "residual_evaluation_ratio_reported": True,
    }
)
CONTRACT["trajectory_gates"].pop(
    "minimum_warm_roots_without_exact_refresh", None
)
CONTRACT["trajectory_gates"].update(
    {
        "warm_refresh_count_is_diagnostic_not_binding": True,
        "same_history_warm_to_cold_wall_ratio_maximum": 0.75,
        "accepted_physical_time_per_wall_hour_reported": True,
    }
)
CONTRACT["execution_order"] = [
    "validate_manifest_parent_seed_sources_and_thread_environment",
    "load_and_bitwise_roundtrip_canonical_seed",
    "run_cold_1_then_three_iteration_reserve_warm_roots_fail_closed",
    "restart_after_warm_1_and_replay_warm_2_then_warm_3_bitwise",
    "run_same_history_cold_shadow_for_warm_2_without_propagation",
    "run_two_cold_half_steps_from_warm_3_start_without_propagation",
    "classify_scientific_validity_before_wall_time_cost",
    "canonicalize_all_pass_or_failure_evidence",
]


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
    seed_hashes = _checksums(SEED_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    if (
        summary["classification"]
        != "warm_policy_scientific_and_cost_passed"
        or not summary["passed"]
        or not summary["scientific_passed"]
        or not summary["cost_passed"]
        or not summary["full_primary_retry_manifest_authorized"]
        or summary["full_primary_retry_execution_authorized"]
        or summary["parent_classification_preserved"]
        != "bounded_continuation_failed"
        or not metrics["endpoint_agreement"]["passed"]
        or metrics["cost"]["warm_to_same_history_cold_wall_ratio"] > 0.75
    ):
        raise RuntimeError("warm-policy certificate authorization changed")
    return {
        "summary": summary,
        "decisive_hashes": {
            name: hashes[name]
            for name in (
                "summary.json",
                "metrics.json",
                "result_warm_1.npz",
                "checkpoint_warm_1.npz",
                "provenance.json",
            )
        },
        "seed_package_hashes": seed_hashes,
    }


def _seed_lock() -> dict:
    return e14c._seed_lock()


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
    seed = _seed_lock()
    if not _tracked_tree_is_clean():
        raise RuntimeError("primary retry manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "iteration_reserve_primary_retry_manifest_frozen_execution_"
            "authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "parent_classification_preserved": "bounded_continuation_failed",
        "primary_bounded_continuation_execution_authorized": True,
        "heldout_continuation_authorized": False,
        "operational_timestep_study_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next_action": "execute_only_the_frozen_primary_retry",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "execution_manifest.json", CONTRACT)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", parent)
    _write(ARTIFACT_DIRECTORY / "seed_lock.json", seed)
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
            "parent_summary_sha256": _sha(PARENT_DIRECTORY / "summary.json"),
            "canonical_seed_sha256": seed["sha256"],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment_at_freeze": {
                name: os.environ.get(name)
                for name in CONTRACT["profiling_contract"]["thread_environment"]
            },
        },
    )
    names = (
        "execution_manifest.json",
        "parent_lock.json",
        "provenance.json",
        "seed_lock.json",
        "summary.json",
    )
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
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
