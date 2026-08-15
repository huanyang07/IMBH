#!/usr/bin/env python3
"""Freeze and execute restartable stages of the remaining fixed-Q ladder."""

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

import run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1 as e1  # noqa: E402
import run_causal_inner_face36_fixed_q_primary_case_recovery_wp10c9d6c7c3b5c4f24e6 as e6  # noqa: E402


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e9"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_remaining_history_ladder_manifest_"
    "wp10c9d6c7c3b5c4f24e9"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_remaining_history_ladder_"
    "wp10c9d6c7c3b5c4f24e9"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / RESULT_ARTIFACT
PRIMARY_DIRECTORY = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e8"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_remaining_history_ladder_"
    "wp10c9d6c7c3b5c4f24e9.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_remaining_history_ladder_"
    "wp10c9d6c7c3b5c4f24e9.py"
)
REMAINING_CASES = e1.CASE_ORDER[1:]
CONTRACT = {
    "schema_version": 1,
    "reused_case": "primary_coarse",
    "remaining_cases": list(REMAINING_CASES),
    "case_order": list(e1.CASE_ORDER),
    "timesteps_seconds": list(e1.TIMESTEPS),
    "binding_temporal_form": "exact_increment_primary",
    "required_schur_solve_method": "row_column_equilibrated_LU_refined_1",
    "gates": e1.GATES,
    "require_bitwise_restart_roundtrip": True,
    "require_bitwise_BDF2_replay": True,
    "minimum_state_rate_convergence_order": 0.9,
    "minimum_reaction_action_convergence_order": 0.9,
    "may_change_physical_equations": False,
    "may_relax_any_gate": False,
    "one_Q_execution_manifest_authorized": False,
    "fixed_Q_micro_solver_authorized": False,
    "reduced_slow_evolution_authorized": False,
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


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
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


def _primary_case() -> dict:
    summary = _read(PRIMARY_DIRECTORY / "summary.json")
    metrics = _read(PRIMARY_DIRECTORY / "metrics.json")
    if (
        not summary["passed"]
        or summary["classification"]
        != "fixed_Q_primary_case_recovered_remaining_history_manifest_authorized"
        or not metrics["case"]["passed"]
        or not metrics["case"]["restart_roundtrip_bitwise"]
        or not metrics["case"]["BDF2_replay_bitwise"]
    ):
        raise RuntimeError("certified primary fixed-Q case changed")
    return metrics["case"]


def _stage_directory(case: str) -> Path:
    return ROOT / "results/canonical" / (
        "causal_inner_face36_fixed_q_remaining_history_ladder_stage_"
        f"{case}_wp10c9d6c7c3b5c4f24e9"
    )


def _configure() -> None:
    e1.WORK_PACKAGE = WORK_PACKAGE
    e1.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
    e1.IMPLEMENTATION_ARTIFACT = PRIMARY_DIRECTORY
    e1.THIS_RUNNER = THIS_RUNNER
    original_identity = e1._identity
    original_metrics = e1._result_metrics

    def identity() -> dict:
        payload = original_identity()
        payload.update(
            {
                "execution_test_sha256": _sha(ROOT / THIS_TEST),
                "manifest_summary_sha256": _sha(
                    MANIFEST_DIRECTORY / "summary.json"
                ),
                "primary_summary_sha256": _sha(
                    PRIMARY_DIRECTORY / "summary.json"
                ),
                "primary_metrics_sha256": _sha(
                    PRIMARY_DIRECTORY / "metrics.json"
                ),
            }
        )
        return payload

    def result_metrics(result, data) -> dict:
        payload = original_metrics(result, data)
        monolithic = result.evaluation.monolithic_evaluation
        payload.update(
            {
                "binding_uses_exact_primitive_increment": bool(
                    monolithic.temporal_storage_uses_exact_primitive_increment
                ),
                "binding_uses_direct_rate_action": bool(
                    monolithic.temporal_storage_uses_direct_rate_action
                ),
                "direct_audit_uses_direct_rate_action": bool(
                    result.direct_rate_evaluation.monolithic_evaluation
                    .temporal_storage_uses_direct_rate_action
                ),
                "mapped_endpoint_path_closure_defect": float(
                    monolithic.maximum_mapped_endpoint_path_closure_defect
                ),
            }
        )
        return payload

    e1._identity = identity
    e1._result_metrics = result_metrics


def _seed_prior_cases(case: str) -> None:
    position = e1.CASE_ORDER.index(case)
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for prior in e1.CASE_ORDER[:position]:
        target = CHECKPOINT_DIRECTORY / f"{prior}.json"
        if target.exists():
            continue
        if prior == "primary_coarse":
            payload = _primary_case()
        else:
            payload = _read(_stage_directory(prior) / "metrics.json")
        _write(target, payload)


def _freeze() -> dict:
    _primary_case()
    if not _tracked_tree_is_clean():
        raise RuntimeError("remaining-ladder manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_remaining_history_ladder_manifest_frozen_"
            "fail_fast_execution_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "next_case": "heldout_coarse",
        "remaining_ladder_execution_authorized": True,
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
            "primary_summary_sha256": _sha(PRIMARY_DIRECTORY / "summary.json"),
            "primary_metrics_sha256": _sha(PRIMARY_DIRECTORY / "metrics.json"),
        },
    )
    names = ("execution_manifest.json", "provenance.json", "summary.json")
    (MANIFEST_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(MANIFEST_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    e6.WORK_PACKAGE = WORK_PACKAGE
    e6._catalog(MANIFEST_DIRECTORY, MANIFEST_ARTIFACT, summary, "PROSPECTIVE")
    return summary


def _canonicalize_stage(case: str, metrics: dict) -> dict:
    directory = _stage_directory(case)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            f"fixed_Q_remaining_history_stage_{case}_passed"
            if metrics["passed"]
            else f"fixed_Q_remaining_history_stage_{case}_failed"
        ),
        "passed": bool(metrics["passed"]),
        "case": case,
        "next_case": (
            e1.CASE_ORDER[e1.CASE_ORDER.index(case) + 1]
            if metrics["passed"] and case != e1.CASE_ORDER[-1]
            else None
        ),
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    arrays = {}
    for stage in ("bdf1", "bdf2"):
        path = CHECKPOINT_DIRECTORY / f"{case}_{stage}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as source:
                for name in source.files:
                    if name != "metrics_json":
                        arrays[f"{stage}_{name}"] = np.asarray(source[name])
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "contract.json", CONTRACT)
    _write(directory / "metrics.json", metrics)
    _write(directory / "summary.json", summary)
    _write_npz(directory / "decisive_arrays.npz", **arrays)
    _write(
        directory / "provenance.json",
        {
            "schema_version": 1,
            **e1._identity(),
            "case": case,
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
    (directory / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(directory / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    e6.WORK_PACKAGE = WORK_PACKAGE
    e6._catalog(
        directory,
        directory.name,
        summary,
        "SUPPORTED" if metrics["passed"] else "REJECTED",
    )
    return summary


def _execute_case(case: str) -> dict:
    if case not in REMAINING_CASES:
        raise ValueError("case is not in the remaining fixed-Q ladder")
    if not _read(MANIFEST_DIRECTORY / "summary.json")[
        "remaining_ladder_execution_authorized"
    ]:
        raise RuntimeError("remaining fixed-Q ladder is not authorized")
    if not _tracked_tree_is_clean():
        raise RuntimeError("remaining-ladder execution requires a clean tree")
    _configure()
    _seed_prior_cases(case)
    metrics = e1._solve_case(case)
    summary = _canonicalize_stage(case, metrics)
    return {"summary": summary, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--case", choices=REMAINING_CASES)
    arguments = parser.parse_args()
    if arguments.freeze == (arguments.case is not None):
        raise SystemExit("select exactly one --freeze or --case")
    payload = _freeze() if arguments.freeze else _execute_case(arguments.case)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
