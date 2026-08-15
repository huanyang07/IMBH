#!/usr/bin/env python3
"""Freeze and execute the adaptive-refresh refined fixed-Q history ladder."""

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


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e13"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_manifest_"
    "wp10c9d6c7c3b5c4f24e13"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_"
    "wp10c9d6c7c3b5c4f24e13"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / RESULT_ARTIFACT
PRIMARY_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
HELDOUT_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_heldout_"
    "wp10c9d6c7c3b5c4f24e12"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_adaptive_refresh_refined_"
    "ladder_wp10c9d6c7c3b5c4f24e13.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_adaptive_refresh_refined_"
    "ladder_wp10c9d6c7c3b5c4f24e13.py"
)
REUSED_CASES = ("primary_coarse", "heldout_coarse")
REFINED_CASES = (
    "primary_middle",
    "heldout_middle",
    "primary_fine",
    "heldout_fine",
)
CONTRACT = {
    "schema_version": 1,
    "reused_cases": list(REUSED_CASES),
    "refined_cases": list(REFINED_CASES),
    "case_order": list(e1.CASE_ORDER),
    "timesteps_seconds": list(e1.TIMESTEPS),
    "binding_temporal_form": "exact_increment_primary",
    "direct_rate_form": "post_root_parity_audit_only",
    "exact_jacobian_refresh_policy": "on_line_search_failure",
    "maximum_exact_jacobian_assemblies_per_root": 2,
    "required_schur_solve_method": "row_column_equilibrated_LU_refined_1",
    "require_all_existing_step_acceptance_gates": True,
    "require_bitwise_restart_roundtrip": True,
    "require_bitwise_BDF2_replay": True,
    "minimum_state_rate_convergence_order": 0.9,
    "minimum_reaction_action_convergence_order": 0.9,
    "fail_fast": True,
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


def _parents() -> tuple[dict, dict]:
    primary = _read(PRIMARY_DIRECTORY / "summary.json")
    heldout = _read(HELDOUT_DIRECTORY / "summary.json")
    if (
        not primary["passed"]
        or not primary["heldout_retry_manifest_authorized"]
        or not heldout["passed"]
        or not heldout["refined_ladder_manifest_authorized"]
    ):
        raise RuntimeError("refined adaptive-refresh ladder is not authorized")
    return primary, heldout


def _stage_directory(case: str) -> Path:
    return ROOT / "results/canonical" / (
        "causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_stage_"
        f"{case}_wp10c9d6c7c3b5c4f24e13"
    )


def _configure() -> None:
    e11._configure()
    e1.WORK_PACKAGE = WORK_PACKAGE
    e1.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
    e1.THIS_RUNNER = THIS_RUNNER
    e1.GATES = {**e1.GATES, "maximum_complete_Jacobian_assemblies": 2}
    original_identity = e1._identity
    original_metrics = e1._result_metrics

    def identity() -> dict:
        payload = original_identity()
        payload.update(
            {
                "refined_ladder_runner_sha256": _sha(ROOT / THIS_RUNNER),
                "refined_ladder_test_sha256": _sha(ROOT / THIS_TEST),
                "refined_ladder_manifest_summary_sha256": _sha(
                    MANIFEST_DIRECTORY / "summary.json"
                ),
                "primary_coarse_summary_sha256": _sha(
                    PRIMARY_DIRECTORY / "summary.json"
                ),
                "heldout_coarse_summary_sha256": _sha(
                    HELDOUT_DIRECTORY / "summary.json"
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


def _coarse_metrics(case: str) -> dict:
    directory = PRIMARY_DIRECTORY if case == "primary_coarse" else HELDOUT_DIRECTORY
    metrics = _read(directory / "metrics.json")
    if not metrics["passed"] or not metrics["BDF2_replay_bitwise"]:
        raise RuntimeError(f"certified {case} evidence changed")
    return metrics


def _seed_prior_cases(case: str | None = None) -> None:
    limit = len(e1.CASE_ORDER) if case is None else e1.CASE_ORDER.index(case)
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for prior in e1.CASE_ORDER[:limit]:
        target = CHECKPOINT_DIRECTORY / f"{prior}.json"
        if target.exists():
            continue
        if prior in REUSED_CASES:
            payload = _coarse_metrics(prior)
        else:
            payload = _read(_stage_directory(prior) / "metrics.json")
        _write(target, payload)


def _freeze() -> dict:
    primary, heldout = _parents()
    if not _tracked_tree_is_clean():
        raise RuntimeError("refined-ladder manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "adaptive_refresh_refined_ladder_manifest_frozen_"
            "fail_fast_execution_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "next_case": "primary_middle",
        "refined_ladder_execution_authorized": True,
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
                ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py"
            ),
            "primary_summary_sha256": _sha(PRIMARY_DIRECTORY / "summary.json"),
            "primary_arrays_sha256": _sha(
                PRIMARY_DIRECTORY / "decisive_arrays.npz"
            ),
            "heldout_summary_sha256": _sha(HELDOUT_DIRECTORY / "summary.json"),
            "heldout_arrays_sha256": _sha(
                HELDOUT_DIRECTORY / "decisive_arrays.npz"
            ),
            "primary_classification": primary["classification"],
            "heldout_classification": heldout["classification"],
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


def _canonicalize_stage(case: str, metrics: dict) -> dict:
    directory = _stage_directory(case)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            f"adaptive_refresh_refined_ladder_stage_{case}_passed"
            if metrics["passed"]
            else f"adaptive_refresh_refined_ladder_stage_{case}_failed"
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
    (directory / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(directory / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    e2._catalog(
        directory,
        directory.name,
        summary,
        "SUPPORTED" if metrics["passed"] else "REJECTED",
    )
    return summary


def _execute_case(case: str) -> dict:
    if case not in REFINED_CASES:
        raise ValueError("case is not in the refined adaptive-refresh ladder")
    if not _read(MANIFEST_DIRECTORY / "summary.json")[
        "refined_ladder_execution_authorized"
    ]:
        raise RuntimeError("refined adaptive-refresh ladder is not authorized")
    if not _tracked_tree_is_clean():
        raise RuntimeError("refined-ladder execution requires a clean tree")
    _configure()
    _seed_prior_cases(case)
    metrics = e1._solve_case(case)
    summary = _canonicalize_stage(case, metrics)
    return {"summary": summary, "metrics": metrics}


def _finalize() -> dict:
    if not _tracked_tree_is_clean():
        raise RuntimeError("refined-ladder finalization requires a clean tree")
    _configure()
    _seed_prior_cases()
    summary = e1._finalize()
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(RESULT_DIRECTORY / "contract.json", CONTRACT)
    _write(RESULT_DIRECTORY / "summary.json", summary)
    _write(
        RESULT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            **e1._identity(),
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
        },
    )
    names = ("contract.json", "provenance.json", "summary.json")
    (RESULT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(RESULT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    e2._catalog(
        RESULT_DIRECTORY,
        RESULT_ARTIFACT,
        summary,
        "SUPPORTED" if summary["passed"] else "REJECTED",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--case", choices=REFINED_CASES)
    parser.add_argument("--finalize", action="store_true")
    arguments = parser.parse_args()
    selected = int(arguments.freeze) + int(arguments.case is not None) + int(
        arguments.finalize
    )
    if selected != 1:
        raise SystemExit("select exactly one --freeze, --case, or --finalize")
    if arguments.freeze:
        payload = _freeze()
    elif arguments.finalize:
        payload = _finalize()
    else:
        payload = _execute_case(arguments.case)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
