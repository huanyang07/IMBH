#!/usr/bin/env python3
"""Aggregate accepted primary continuation evidence without rerunning roots."""

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14r"
ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_evidence_aggregation_"
    "wp10c9d6c7c3b5c4f24e14r"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
MANIFEST_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_evidence_aggregation_manifest_"
    "wp10c9d6c7c3b5c4f24e14q"
)
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


def _validate_manifest() -> dict:
    manifest_hashes = _checksums(MANIFEST_DIRECTORY)
    summary = _read(MANIFEST_DIRECTORY / "summary.json")
    contract = _read(MANIFEST_DIRECTORY / "aggregation_contract.json")
    parent = _read(MANIFEST_DIRECTORY / "parent_lock.json")
    provenance = _read(MANIFEST_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["primary_evidence_aggregation_authorized"]
        or summary["heldout_continuation_manifest_authorized"]
        or not contract["aggregation_only"]
        or contract["physical_root_execution_authorized"]
        or contract["historical_parent_classification_preserved"]
        != "bounded_continuation_failed"
    ):
        raise RuntimeError("e14q aggregation authorization changed")
    for relative, digest in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != digest:
            raise RuntimeError(f"frozen aggregation source changed: {relative}")
    directories = {
        "historical_retry": RETRY_DIRECTORY,
        "positive_diagnosis": DIAGNOSIS_DIRECTORY,
        "policy_certificate": POLICY_DIRECTORY,
    }
    for label, directory in directories.items():
        actual = _checksums(directory)
        if actual != parent["package_hashes"][label]:
            raise RuntimeError(f"frozen parent package changed: {label}")
    return {
        "summary": summary,
        "contract": contract,
        "parent": parent,
        "provenance": provenance,
        "manifest_hashes": manifest_hashes,
    }


def _all_acceptance_gates(root: dict) -> bool:
    acceptance = root.get("acceptance", {})
    root_passed = root.get(
        "root_passed", acceptance.get("nonlinear_root_passed", False)
    )
    return bool(
        root.get("accepted")
        and root_passed
        and acceptance.get("accepted")
        and not acceptance.get("failure_reasons")
        and all(value for key, value in acceptance.items() if key.endswith("_passed"))
    )


def _evaluate_evidence(
    retry_summary: dict,
    retry_metrics: dict,
    diagnosis_summary: dict,
    diagnosis_metrics: dict,
    policy_summary: dict,
    policy_metrics: dict,
    contract: dict,
) -> dict:
    thresholds = contract["binding_primary_evidence"]
    roots = retry_metrics["main_roots"]
    labels = ("cold_1", "warm_1", "warm_2", "warm_3")
    replay = retry_metrics["replay"]
    half = retry_metrics["matched_endpoint_half_step_audit"]
    shadow = retry_metrics["same_history_cold_shadow"]
    policy_audit = policy_metrics["polished_candidate_audit"]
    replay_roots = replay.get("roots", {})
    gates = {
        "historical_classification_preserved": (
            retry_summary["classification"] == "bounded_continuation_failed"
            and retry_metrics["classification"] == "bounded_continuation_failed"
        ),
        "main_root_count": (
            retry_summary["accepted_main_BDF2_roots"]
            == thresholds["accepted_main_BDF2_roots"]
            and retry_summary["rejected_main_BDF2_roots"] == 0
            and retry_summary["planned_ladder_complete"]
        ),
        "accepted_horizon": (
            retry_summary["accepted_main_horizon_seconds"]
            == thresholds["accepted_main_horizon_seconds"]
        ),
        "all_main_roots_accepted": (
            tuple(roots) == labels and all(_all_acceptance_gates(roots[x]) for x in labels)
        ),
        "suffix_replay_bitwise": bool(
            replay.get("executed")
            and replay.get("passed")
            and all(
                replay_roots.get(label, {}).get("accepted")
                and replay_roots[label].get("result_bitwise")
                and replay_roots[label].get("continuation_bitwise")
                for label in ("warm_2", "warm_3")
            )
        ),
        "cumulative_ledger": bool(
            retry_metrics["cumulative_ledger_complete"]
            and retry_metrics["cumulative_ledger_passed"]
        ),
        "matched_endpoint_half_step": bool(
            half.get("executed")
            and half.get("passed")
            and _all_acceptance_gates(half["half_1"])
            and _all_acceptance_gates(half["half_2"])
        ),
        "same_history_control_accepted": bool(
            shadow.get("executed") and _all_acceptance_gates(shadow["root"])
        ),
        "same_history_cost": bool(
            shadow.get("cost_passed")
            and shadow["warm_to_cold_wall_time_ratio"]
            <= thresholds["maximum_warm_to_cold_wall_time_ratio"]
            and shadow["warm_to_cold_residual_evaluation_ratio"]
            <= thresholds["maximum_warm_to_cold_residual_evaluation_ratio"]
        ),
        "positive_diagnosis": bool(
            diagnosis_summary.get("passed")
            and diagnosis_summary.get("classification")
            == "cold_shadow_residual_limited_action_equivalence_diagnosed"
            and diagnosis_metrics.get("positive_diagnosis")
        ),
        "certified_comparison_policy": bool(
            policy_summary.get("passed")
            and policy_summary.get("classification")
            == "same_history_equivalence_policy_certified"
            and not policy_summary.get("production_step_acceptance_changed")
            and policy_metrics.get("policy_passed")
            and policy_metrics["polished_candidate_audit"]["maximum_scaled_residual"]
            <= thresholds["certified_control_comparison_residual"]
            and policy_metrics["polished_to_warm_comparison"][
                "scaled_state_absolute_defect"
            ]
            <= thresholds["maximum_scaled_state_difference"]
            and policy_metrics["polished_to_warm_comparison"][
                "reaction_action_relative_defect"
            ]
            <= thresholds["maximum_reaction_action_relative_difference"]
            and policy_audit.get("passed")
            and all(policy_audit["checks"].values())
        ),
        "nonpropagation": bool(
            not diagnosis_summary.get("continuation_state_constructed")
            and not policy_summary.get("continuation_state_constructed")
            and not policy_summary.get("trajectory_executed")
            and policy_metrics["nonpropagation"][
                "accepted_trajectory_horizon_seconds_added"
            ]
            == 0.0
            and not policy_metrics["nonpropagation"]["candidate_entered_history"]
        ),
    }
    return {"gates": gates, "passed": all(gates.values())}


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PASS" if summary["passed"] else "FAIL",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
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


def _run() -> dict:
    manifest = _validate_manifest()
    if not _tracked_tree_is_clean():
        raise RuntimeError("primary evidence aggregation requires a clean tree")
    retry_summary = _read(RETRY_DIRECTORY / "summary.json")
    retry_metrics = _read(RETRY_DIRECTORY / "metrics.json")
    diagnosis_summary = _read(DIAGNOSIS_DIRECTORY / "summary.json")
    diagnosis_metrics = _read(DIAGNOSIS_DIRECTORY / "metrics.json")
    policy_summary = _read(POLICY_DIRECTORY / "summary.json")
    policy_metrics = _read(POLICY_DIRECTORY / "metrics.json")
    evaluation = _evaluate_evidence(
        retry_summary,
        retry_metrics,
        diagnosis_summary,
        diagnosis_metrics,
        policy_summary,
        policy_metrics,
        manifest["contract"],
    )
    passed = evaluation["passed"]
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "primary_bounded_continuation_evidence_certified"
            if passed
            else "primary_evidence_aggregation_failed"
        ),
        "passed": passed,
        "aggregation_executed": True,
        "trajectory_executed": False,
        "historical_parent_classification_preserved": "bounded_continuation_failed",
        "accepted_primary_BDF2_roots": 4 if passed else 0,
        "accepted_primary_horizon_seconds": 4.0e-7 if passed else 0.0,
        "heldout_continuation_manifest_authorized": passed,
        "heldout_continuation_execution_authorized": False,
        "operational_timestep_study_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    metrics = {
        "schema_version": 1,
        "evaluation": evaluation,
        "accepted_primary_evidence": {
            "main_roots": {
                label: {
                    "maximum_scaled_residual": retry_metrics["main_roots"][label][
                        "maximum_scaled_residual"
                    ],
                    "function_evaluations": retry_metrics["main_roots"][label][
                        "function_evaluations"
                    ],
                    "exact_Jacobian_assemblies": retry_metrics["main_roots"][label][
                        "exact_Jacobian_assemblies"
                    ],
                }
                for label in ("cold_1", "warm_1", "warm_2", "warm_3")
            },
            "accepted_horizon_seconds": retry_summary[
                "accepted_main_horizon_seconds"
            ],
            "cumulative_absolute_ledger_defect": retry_metrics[
                "cumulative_absolute_ledger_defect"
            ],
            "suffix_replay_passed": retry_metrics["replay"]["passed"],
            "half_step_state_difference": retry_metrics[
                "matched_endpoint_half_step_audit"
            ]["state_difference_relative_to_full_step_change"],
            "half_step_reaction_action_difference": retry_metrics[
                "matched_endpoint_half_step_audit"
            ]["reaction_action_relative_difference"],
            "warm_to_cold_wall_time_ratio": retry_metrics[
                "same_history_cold_shadow"
            ]["warm_to_cold_wall_time_ratio"],
            "warm_to_cold_residual_evaluation_ratio": retry_metrics[
                "same_history_cold_shadow"
            ]["warm_to_cold_residual_evaluation_ratio"],
            "polished_state_difference": policy_metrics[
                "polished_to_warm_comparison"
            ]["scaled_state_absolute_defect"],
            "polished_reaction_action_difference": policy_metrics[
                "polished_to_warm_comparison"
            ]["reaction_action_relative_defect"],
        },
        "nonpropagation": {
            "physical_roots_rerun": 0,
            "trajectory_horizon_seconds_added": 0.0,
        },
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(CANONICAL_DIRECTORY / "evidence_ledger.json", metrics)
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "manifest_summary_sha256": _sha(MANIFEST_DIRECTORY / "summary.json"),
            "source_hashes": manifest["provenance"]["source_hashes"],
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
    files = ("evidence_ledger.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in files),
        encoding="utf-8",
    )
    _catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        raise SystemExit("select --run")
    print(json.dumps(_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
