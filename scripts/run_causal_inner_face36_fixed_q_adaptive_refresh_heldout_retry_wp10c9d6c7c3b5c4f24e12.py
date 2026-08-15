#!/usr/bin/env python3
"""Freeze and execute the adaptive-refresh held-out coarse retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_fixed_q_adaptive_refresh_primary_revalidation_wp10c9d6c7c3b5c4f24e11 as e11  # noqa: E402
import run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1 as e1  # noqa: E402
import run_causal_inner_face36_fixed_q_exact_refresh_diagnostic_wp10c9d6c7c3b5c4f24e2 as e2  # noqa: E402


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e12"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_adaptive_refresh_heldout_manifest_"
    "wp10c9d6c7c3b5c4f24e12"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_adaptive_refresh_heldout_"
    "wp10c9d6c7c3b5c4f24e12"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / RESULT_ARTIFACT
PRIMARY_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
REJECTED_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_remaining_history_ladder_stage_"
    "heldout_coarse_wp10c9d6c7c3b5c4f24e9"
)
DIAGNOSTIC_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_heldout_bdf2_exact_refresh_"
    "wp10c9d6c7c3b5c4f24e10"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_adaptive_refresh_heldout_"
    "retry_wp10c9d6c7c3b5c4f24e12.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_adaptive_refresh_heldout_"
    "retry_wp10c9d6c7c3b5c4f24e12.py"
)
CONTRACT = {
    "schema_version": 1,
    "case": "heldout_coarse",
    "state": "heldout_16ms",
    "timestep_seconds": 1.0e-7,
    "binding_temporal_form": "exact_increment_primary",
    "exact_jacobian_refresh_policy": "on_line_search_failure",
    "maximum_exact_jacobian_assemblies": 2,
    "required_BDF1_exact_jacobian_assemblies": 1,
    "required_BDF2_exact_jacobian_assemblies": 2,
    "required_BDF2_function_evaluations": 18,
    "required_optional_refresh_reason": "line_search_failure",
    "maximum_scaled_residual": 1.0e-10,
    "require_all_existing_step_acceptance_gates": True,
    "require_BDF1_bitwise_historical_arrays": True,
    "require_bitwise_restart_roundtrip": True,
    "require_bitwise_BDF2_replay": True,
    "historical_rejection_must_be_preserved": True,
    "may_resume_refined_ladder": False,
    "may_change_physical_equations": False,
    "may_relax_any_gate": False,
}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _parents() -> tuple[dict, dict, dict]:
    primary = _read(PRIMARY_DIRECTORY / "summary.json")
    rejected = _read(REJECTED_DIRECTORY / "summary.json")
    diagnostic = _read(DIAGNOSTIC_DIRECTORY / "summary.json")
    if (
        not primary["passed"]
        or not primary["heldout_retry_manifest_authorized"]
        or rejected["passed"]
        or not diagnostic["passed"]
        or not diagnostic["adaptive_refresh_policy_manifest_authorized"]
    ):
        raise RuntimeError("held-out retry authorization changed")
    return primary, rejected, diagnostic


def _freeze() -> dict:
    primary, rejected, diagnostic = _parents()
    if not _tracked_tree_is_clean():
        raise RuntimeError("held-out retry manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "adaptive_refresh_heldout_retry_manifest_frozen_execution_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "heldout_retry_authorized": True,
        "remaining_ladder_execution_authorized": False,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    MANIFEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(MANIFEST_DIRECTORY / "execution_manifest.json", CONTRACT)
    _write(MANIFEST_DIRECTORY / "summary.json", summary)
    _write(
        MANIFEST_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner_sha256": _sha(ROOT / THIS_RUNNER),
            "test_sha256": _sha(ROOT / THIS_TEST),
            "fixed_q_source_sha256": _sha(
                ROOT
                / "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py"
            ),
            "primary_summary_sha256": _sha(PRIMARY_DIRECTORY / "summary.json"),
            "primary_arrays_sha256": _sha(
                PRIMARY_DIRECTORY / "decisive_arrays.npz"
            ),
            "rejected_summary_sha256": _sha(
                REJECTED_DIRECTORY / "summary.json"
            ),
            "rejected_arrays_sha256": _sha(
                REJECTED_DIRECTORY / "decisive_arrays.npz"
            ),
            "diagnostic_summary_sha256": _sha(
                DIAGNOSTIC_DIRECTORY / "summary.json"
            ),
            "primary_classification": primary["classification"],
            "rejected_classification": rejected["classification"],
            "diagnostic_classification": diagnostic["classification"],
        },
    )
    names = ("execution_manifest.json", "provenance.json", "summary.json")
    (MANIFEST_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(MANIFEST_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    e2._catalog(MANIFEST_DIRECTORY, MANIFEST_ARTIFACT, summary, "PROSPECTIVE")
    return summary


def _configure() -> None:
    e11._configure()
    e1.WORK_PACKAGE = WORK_PACKAGE
    e1.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
    original_identity = e1._identity
    original_metrics = e1._result_metrics

    def identity() -> dict:
        payload = original_identity()
        payload.update(
            {
                "heldout_retry_runner_sha256": _sha(ROOT / THIS_RUNNER),
                "heldout_retry_test_sha256": _sha(ROOT / THIS_TEST),
                "heldout_retry_manifest_summary_sha256": _sha(
                    MANIFEST_DIRECTORY / "summary.json"
                ),
                "primary_result_summary_sha256": _sha(
                    PRIMARY_DIRECTORY / "summary.json"
                ),
            }
        )
        return payload

    def result_metrics(result, data) -> dict:
        payload = original_metrics(result, data)
        payload["maximum_exact_Jacobian_assemblies_allowed"] = 2
        return payload

    e1._identity = identity
    e1._result_metrics = result_metrics
    e1.GATES = {**e1.GATES, "maximum_complete_Jacobian_assemblies": 2}


def _seed_primary() -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    primary_metrics = _read(PRIMARY_DIRECTORY / "metrics.json")
    _write(CHECKPOINT_DIRECTORY / "primary_coarse.json", primary_metrics)


def _checkpoint_arrays() -> dict[str, np.ndarray]:
    arrays = {}
    for stage in ("bdf1", "bdf2"):
        path = CHECKPOINT_DIRECTORY / f"heldout_coarse_{stage}.npz"
        with np.load(path, allow_pickle=False) as source:
            for name in source.files:
                if name != "metrics_json":
                    arrays[f"{stage}_{name}"] = np.asarray(source[name])
    return arrays


def _historical_bdf1_bitwise(arrays: dict[str, np.ndarray]) -> bool:
    with np.load(
        REJECTED_DIRECTORY / "decisive_arrays.npz", allow_pickle=False
    ) as historical:
        names = [name for name in arrays if name.startswith("bdf1_")]
        return bool(
            all(name in historical.files for name in names)
            and all(
                np.array_equal(arrays[name], historical[name]) for name in names
            )
        )


def _execute() -> dict:
    _parents()
    manifest = _read(MANIFEST_DIRECTORY / "summary.json")
    if not manifest["heldout_retry_authorized"]:
        raise RuntimeError("adaptive held-out retry is not authorized")
    if not _tracked_tree_is_clean():
        raise RuntimeError("adaptive held-out retry requires a clean tree")
    _configure()
    _seed_primary()
    metrics = e1._solve_case("heldout_coarse")
    arrays = _checkpoint_arrays()
    bdf1_bitwise = _historical_bdf1_bitwise(arrays)
    bdf1_budget = bool(metrics["BDF1"]["exact_Jacobian_assemblies"] == 1)
    bdf2_budget = bool(metrics["BDF2"]["exact_Jacobian_assemblies"] == 2)
    evaluation_count = bool(
        metrics["BDF2"]["function_evaluations"]
        == CONTRACT["required_BDF2_function_evaluations"]
    )
    passed = bool(
        metrics["passed"]
        and metrics["restart_roundtrip_bitwise"]
        and metrics["BDF2_replay_bitwise"]
        and bdf1_bitwise
        and bdf1_budget
        and bdf2_budget
        and evaluation_count
    )
    metrics.update(
        {
            "historical_BDF1_decisive_arrays_bitwise": bdf1_bitwise,
            "BDF1_exact_Jacobian_budget_passed": bdf1_budget,
            "BDF2_exact_Jacobian_budget_passed": bdf2_budget,
            "BDF2_function_evaluation_contract_passed": evaluation_count,
            "adaptive_heldout_retry_passed": passed,
        }
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "adaptive_refresh_heldout_coarse_passed_"
            "refined_ladder_manifest_authorized"
            if passed
            else "adaptive_refresh_heldout_coarse_retry_failed"
        ),
        "passed": passed,
        "historical_rejection_preserved": True,
        "historical_BDF1_decisive_arrays_bitwise": bdf1_bitwise,
        "adaptive_refresh_used": bdf2_budget,
        "refined_ladder_manifest_authorized": passed,
        "remaining_ladder_execution_authorized": False,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(RESULT_DIRECTORY / "contract.json", CONTRACT)
    _write(RESULT_DIRECTORY / "metrics.json", metrics)
    _write(RESULT_DIRECTORY / "summary.json", summary)
    np.savez_compressed(RESULT_DIRECTORY / "decisive_arrays.npz", **arrays)
    _write(
        RESULT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            **e1._identity(),
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
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
    names = (
        "contract.json",
        "decisive_arrays.npz",
        "metrics.json",
        "provenance.json",
        "summary.json",
    )
    (RESULT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(RESULT_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    e2._catalog(
        RESULT_DIRECTORY,
        RESULT_ARTIFACT,
        summary,
        "SUPPORTED" if passed else "REJECTED",
    )
    return {"summary": summary, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.freeze == arguments.execute:
        raise SystemExit("select exactly one of --freeze or --execute")
    payload = _freeze() if arguments.freeze else _execute()
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
