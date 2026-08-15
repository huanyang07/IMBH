#!/usr/bin/env python3
"""Freeze and run the bounded repaired fixed-Q primary-case recovery."""

from __future__ import annotations

import argparse
import csv
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


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e6"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_case_recovery_manifest_"
    "wp10c9d6c7c3b5c4f24e6"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_case_recovery_"
    "wp10c9d6c7c3b5c4f24e6"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / RESULT_ARTIFACT
PARENT_DIRECTORY = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_exact_increment_storage_recertification_"
    "wp10c9d6c7c3b5c4f24e5"
)
CATALOG_CSV = ROOT / "results/manifests/canonical_artifacts.csv"
CATALOG_JSON = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_primary_case_recovery_"
    "wp10c9d6c7c3b5c4f24e6.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_primary_case_recovery_"
    "wp10c9d6c7c3b5c4f24e6.py"
)
CONTRACT = {
    "schema_version": 1,
    "case": "primary_coarse",
    "state": "committed_middle_20ms",
    "timestep_seconds": 1.0e-7,
    "steps": "one_BDF1_start_plus_one_BDF2_step_and_replay",
    "binding_temporal_form": "exact_increment_primary",
    "maximum_scaled_residual": 1.0e-10,
    "maximum_Q3_relative_defect": 1.0e-12,
    "maximum_ledger_relative_defect": 1.0e-12,
    "maximum_storage_parity_relative_defect": 1.0e-9,
    "minimum_path_reconstruction_factor": 1.0 - 1.0e-12,
    "maximum_path_reconstruction_factor": 1.0 + 1.0e-12,
    "maximum_raw_Schur_condition_number": 1.0e8,
    "maximum_H_over_R": 0.12,
    "minimum_scattering_optical_depth": 1.0,
    "maximum_scaled_primitive_change": 5.0e-3,
    "maximum_complete_Jacobian_assemblies_per_root": 1,
    "require_bitwise_restart_roundtrip": True,
    "require_bitwise_BDF2_replay": True,
    "may_change_physical_equations": False,
    "may_change_row_scales_or_merit_norm": False,
    "may_relax_residual_or_physical_gates": False,
    "remaining_history_ladder_execution_authorized": False,
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


def _write(path: Path, payload: dict) -> None:
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


def _catalog(
    directory: Path,
    artifact: str,
    summary: dict,
    status: str,
) -> None:
    with CATALOG_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != artifact]
    for path in sorted(directory.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": artifact,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": status,
                }
            )
    with CATALOG_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CATALOG_JSON)
    catalog["artifacts"][artifact] = {
        "path": str(directory.relative_to(ROOT)),
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
    _write(CATALOG_JSON, catalog)


def _parent_metrics() -> dict:
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    if (
        summary["passed"]
        or summary["classification"]
        != "fixed_Q_exact_increment_residual_resolution_failed"
        or not metrics["gates"]["saved_root"]
        or not metrics["gates"]["full_step_model"]
        or not metrics["gates"]["increment_direct_storage"]
        or not metrics["gates"]["mapped_endpoint_path_closure"]
    ):
        raise RuntimeError("exact-increment endpoint classification changed")
    return metrics


def _freeze() -> dict:
    _parent_metrics()
    if not _tracked_tree_is_clean():
        raise RuntimeError("primary-case manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_primary_case_recovery_manifest_frozen_"
            "bounded_execution_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "primary_case_execution_authorized": True,
        "remaining_history_ladder_execution_authorized": False,
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
            "untracked_files_at_start": _git(
                "ls-files", "--others", "--exclude-standard"
            ).splitlines(),
            "runner_sha256": _sha(ROOT / THIS_RUNNER),
            "test_sha256": _sha(ROOT / THIS_TEST),
            "parent_summary_sha256": _sha(PARENT_DIRECTORY / "summary.json"),
            "parent_metrics_sha256": _sha(PARENT_DIRECTORY / "metrics.json"),
        },
    )
    names = ("execution_manifest.json", "provenance.json", "summary.json")
    (MANIFEST_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(MANIFEST_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _catalog(MANIFEST_DIRECTORY, MANIFEST_ARTIFACT, summary, "PROSPECTIVE")
    return summary


def _execute() -> dict:
    _parent_metrics()
    manifest = _read(MANIFEST_DIRECTORY / "summary.json")
    if not manifest["primary_case_execution_authorized"]:
        raise RuntimeError("primary-case execution is not authorized")
    if not _tracked_tree_is_clean():
        raise RuntimeError("primary-case execution requires a clean tree")
    e1.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
    e1.IMPLEMENTATION_ARTIFACT = PARENT_DIRECTORY
    e1.THIS_RUNNER = THIS_RUNNER
    original_identity = e1._identity
    original_result_metrics = e1._result_metrics

    def identity() -> dict:
        payload = original_identity()
        payload["execution_test_sha256"] = _sha(ROOT / THIS_TEST)
        payload["manifest_summary_sha256"] = _sha(
            MANIFEST_DIRECTORY / "summary.json"
        )
        payload["monolithic_bdf_source_sha256"] = _sha(
            ROOT
            / "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py"
        )
        payload["monolithic_dae_source_sha256"] = _sha(
            ROOT
            / "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py"
        )
        return payload

    def result_metrics(result, data) -> dict:
        payload = original_result_metrics(result, data)
        monolithic = result.evaluation.monolithic_evaluation
        payload["binding_uses_exact_primitive_increment"] = bool(
            monolithic.temporal_storage_uses_exact_primitive_increment
        )
        payload["binding_uses_direct_rate_action"] = bool(
            monolithic.temporal_storage_uses_direct_rate_action
        )
        payload["direct_audit_uses_direct_rate_action"] = bool(
            result.direct_rate_evaluation.monolithic_evaluation
            .temporal_storage_uses_direct_rate_action
        )
        payload["mapped_endpoint_path_closure_defect"] = float(
            monolithic.maximum_mapped_endpoint_path_closure_defect
        )
        return payload

    e1._identity = identity
    e1._result_metrics = result_metrics
    result = e1._solve_case("primary_coarse")
    exact_increment_passed = bool(
        result["BDF1"]["binding_uses_exact_primitive_increment"]
        and not result["BDF1"]["binding_uses_direct_rate_action"]
        and result["BDF1"]["direct_audit_uses_direct_rate_action"]
        and (
            result.get("BDF2") is None
            or (
                result["BDF2"]["binding_uses_exact_primitive_increment"]
                and not result["BDF2"]["binding_uses_direct_rate_action"]
                and result["BDF2"]["direct_audit_uses_direct_rate_action"]
            )
        )
    )
    passed = bool(result["passed"] and exact_increment_passed)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_primary_case_recovered_remaining_history_manifest_authorized"
            if passed
            else "fixed_Q_primary_case_recovery_failed"
        ),
        "passed": passed,
        "bounded_primary_case_only": True,
        "physical_failure_detected": False,
        "parent_rejections_preserved": True,
        "remaining_history_ladder_manifest_authorized": passed,
        "remaining_history_ladder_execution_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    metrics = {
        "case": result,
        "exact_increment_contract_passed": exact_increment_passed,
        "execution_identity": identity(),
    }
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(RESULT_DIRECTORY / "contract.json", CONTRACT)
    _write(RESULT_DIRECTORY / "metrics.json", metrics)
    _write(RESULT_DIRECTORY / "summary.json", summary)
    arrays = {}
    for stage in ("bdf1", "bdf2"):
        path = CHECKPOINT_DIRECTORY / f"primary_coarse_{stage}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as source:
                for name in source.files:
                    if name != "metrics_json":
                        arrays[f"{stage}_{name}"] = np.asarray(source[name])
    _write_npz(RESULT_DIRECTORY / "decisive_arrays.npz", **arrays)
    _write(
        RESULT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            **identity(),
            "tracked_worktree_clean_at_start": True,
            "untracked_files_at_start": _git(
                "ls-files", "--others", "--exclude-standard"
            ).splitlines(),
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
    _catalog(
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
