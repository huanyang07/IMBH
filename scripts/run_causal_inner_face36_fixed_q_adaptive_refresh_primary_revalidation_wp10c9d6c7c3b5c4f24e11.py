#!/usr/bin/env python3
"""Freeze and execute primary coarse non-regression for adaptive refresh."""

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
import run_causal_inner_face36_fixed_q_exact_refresh_diagnostic_wp10c9d6c7c3b5c4f24e2 as e2  # noqa: E402


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e11"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_manifest_"
    "wp10c9d6c7c3b5c4f24e11"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / RESULT_ARTIFACT
DIAGNOSTIC_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_heldout_bdf2_exact_refresh_"
    "wp10c9d6c7c3b5c4f24e10"
)
BASELINE_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_retry_wp10c9d6c7c3b5c4f24e8"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "revalidation_wp10c9d6c7c3b5c4f24e11.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "revalidation_wp10c9d6c7c3b5c4f24e11.py"
)
CONTRACT = {
    "schema_version": 1,
    "case": "primary_coarse",
    "state": "primary_20ms",
    "timestep_seconds": 1.0e-7,
    "binding_temporal_form": "exact_increment_primary",
    "exact_jacobian_refresh_policy": "on_line_search_failure",
    "maximum_exact_jacobian_assemblies": 2,
    "initial_exact_jacobian_required": True,
    "additional_exact_jacobian_allowed_only_after_line_search_failure": True,
    "maximum_scaled_residual": 1.0e-10,
    "require_all_existing_step_acceptance_gates": True,
    "require_bitwise_restart_roundtrip": True,
    "require_bitwise_BDF2_replay": True,
    "require_bitwise_baseline_decisive_arrays": True,
    "primary_nonregression_required_before_heldout_retry": True,
    "may_amend_historical_results": False,
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


def _parents() -> tuple[dict, dict]:
    diagnostic = _read(DIAGNOSTIC_DIRECTORY / "summary.json")
    baseline = _read(BASELINE_DIRECTORY / "summary.json")
    if (
        not diagnostic["passed"]
        or not diagnostic["adaptive_refresh_policy_manifest_authorized"]
        or not baseline["passed"]
        or baseline["classification"]
        != "fixed_Q_primary_case_recovered_remaining_history_manifest_authorized"
    ):
        raise RuntimeError("adaptive-refresh authorization or baseline changed")
    return diagnostic, baseline


def _freeze() -> dict:
    diagnostic, baseline = _parents()
    if not _tracked_tree_is_clean():
        raise RuntimeError("adaptive-refresh manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "adaptive_refresh_policy_manifest_frozen_"
            "primary_revalidation_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "primary_revalidation_authorized": True,
        "heldout_retry_authorized": False,
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
            "diagnostic_summary_sha256": _sha(
                DIAGNOSTIC_DIRECTORY / "summary.json"
            ),
            "diagnostic_metrics_sha256": _sha(
                DIAGNOSTIC_DIRECTORY / "metrics.json"
            ),
            "baseline_summary_sha256": _sha(
                BASELINE_DIRECTORY / "summary.json"
            ),
            "baseline_arrays_sha256": _sha(
                BASELINE_DIRECTORY / "decisive_arrays.npz"
            ),
            "diagnostic_classification": diagnostic["classification"],
            "baseline_classification": baseline["classification"],
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
    e1.WORK_PACKAGE = WORK_PACKAGE
    e1.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
    e1.IMPLEMENTATION_ARTIFACT = DIAGNOSTIC_DIRECTORY
    e1.THIS_RUNNER = THIS_RUNNER
    original_identity = e1._identity
    original_metrics = e1._result_metrics
    original_bdf1 = e1.solve_causal_five_field_fixed_q_backward_euler
    original_bdf = e1.solve_causal_five_field_fixed_q_bdf

    def identity() -> dict:
        payload = original_identity()
        payload.update(
            {
                "execution_test_sha256": _sha(ROOT / THIS_TEST),
                "manifest_summary_sha256": _sha(
                    MANIFEST_DIRECTORY / "summary.json"
                ),
                "baseline_arrays_sha256": _sha(
                    BASELINE_DIRECTORY / "decisive_arrays.npz"
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
                "exact_jacobian_refresh_policy": (
                    "on_line_search_failure"
                ),
            }
        )
        return payload

    def adaptive_bdf1(*args, **kwargs):
        kwargs["maximum_exact_jacobian_refreshes"] = 2
        kwargs["exact_jacobian_refresh_policy"] = "on_line_search_failure"
        return original_bdf1(*args, **kwargs)

    def adaptive_bdf(*args, **kwargs):
        kwargs["maximum_exact_jacobian_refreshes"] = 2
        kwargs["exact_jacobian_refresh_policy"] = "on_line_search_failure"
        return original_bdf(*args, **kwargs)

    e1._identity = identity
    e1._result_metrics = result_metrics
    e1.solve_causal_five_field_fixed_q_backward_euler = adaptive_bdf1
    e1.solve_causal_five_field_fixed_q_bdf = adaptive_bdf


def _checkpoint_arrays() -> dict[str, np.ndarray]:
    arrays = {}
    for stage in ("bdf1", "bdf2"):
        path = CHECKPOINT_DIRECTORY / f"primary_coarse_{stage}.npz"
        with np.load(path, allow_pickle=False) as source:
            for name in source.files:
                if name != "metrics_json":
                    arrays[f"{stage}_{name}"] = np.asarray(source[name])
    return arrays


def _baseline_bitwise(arrays: dict[str, np.ndarray]) -> bool:
    with np.load(
        BASELINE_DIRECTORY / "decisive_arrays.npz", allow_pickle=False
    ) as baseline:
        return bool(
            set(arrays) == set(baseline.files)
            and all(
                np.array_equal(arrays[name], baseline[name]) for name in arrays
            )
        )


def _execute() -> dict:
    _parents()
    manifest = _read(MANIFEST_DIRECTORY / "summary.json")
    if not manifest["primary_revalidation_authorized"]:
        raise RuntimeError("adaptive primary revalidation is not authorized")
    if not _tracked_tree_is_clean():
        raise RuntimeError("adaptive primary revalidation requires a clean tree")
    _configure()
    metrics = e1._solve_case("primary_coarse")
    arrays = _checkpoint_arrays()
    baseline_bitwise = _baseline_bitwise(arrays)
    bdf1_one_initial = bool(metrics["BDF1"]["exact_Jacobian_assemblies"] == 1)
    bdf2_one_initial = bool(metrics["BDF2"]["exact_Jacobian_assemblies"] == 1)
    passed = bool(
        metrics["passed"]
        and metrics["restart_roundtrip_bitwise"]
        and metrics["BDF2_replay_bitwise"]
        and bdf1_one_initial
        and bdf2_one_initial
        and baseline_bitwise
    )
    metrics.update(
        {
            "baseline_decisive_arrays_bitwise": baseline_bitwise,
            "BDF1_used_only_initial_exact_Jacobian": bdf1_one_initial,
            "BDF2_used_only_initial_exact_Jacobian": bdf2_one_initial,
            "adaptive_policy_nonregression_passed": passed,
        }
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "adaptive_refresh_primary_nonregression_passed_"
            "heldout_retry_manifest_authorized"
            if passed
            else "adaptive_refresh_primary_nonregression_failed"
        ),
        "passed": passed,
        "primary_nonregression_passed": passed,
        "baseline_decisive_arrays_bitwise": baseline_bitwise,
        "historical_results_preserved": True,
        "heldout_retry_manifest_authorized": passed,
        "heldout_retry_authorized": False,
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
