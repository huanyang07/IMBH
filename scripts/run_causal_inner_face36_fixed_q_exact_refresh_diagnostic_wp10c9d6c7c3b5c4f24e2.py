#!/usr/bin/env python3
"""Freeze and execute one exact-refresh fixed-Q endpoint diagnostic."""

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
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1 as f24e1  # noqa: E402
from run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a import (  # noqa: E402
    _state_audit,
)

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    solve_causal_five_field_fixed_q_backward_euler,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e2"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_exact_refresh_diagnostic_manifest_"
    "wp10c9d6c7c3b5c4f24e2"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_exact_refresh_diagnostic_"
    "wp10c9d6c7c3b5c4f24e2"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_exact_refresh_diagnostic_"
    "wp10c9d6c7c3b5c4f24e2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_exact_refresh_diagnostic_"
    "wp10c9d6c7c3b5c4f24e2.py"
)
DIAGNOSTIC_CONTRACT = {
    "schema_version": 1,
    "source_case": "primary_coarse",
    "source_stage": "rejected_BDF1_endpoint",
    "timestep_seconds": 1.0e-7,
    "binding_temporal_form": "increment_primary_complete_BDF",
    "reaction_channel_basis": "frozen_normalized",
    "maximum_scaled_residual": 1.0e-10,
    "maximum_Q3_relative_defect": 1.0e-12,
    "maximum_ledger_relative_defect": 1.0e-12,
    "maximum_storage_parity_relative_defect": 1.0e-9,
    "minimum_path_reconstruction_factor": 1.0 - 1.0e-12,
    "maximum_raw_Schur_condition_number": 1.0e8,
    "maximum_H_over_R": 0.12,
    "minimum_scattering_optical_depth": 1.0,
    "maximum_scaled_primitive_change": 5.0e-3,
    "maximum_newton_iterations": 4,
    "maximum_line_search_iterations": 8,
    "exact_Jacobian_assemblies_in_diagnostic": 1,
    "diagnostic_only": True,
    "may_amend_parent_rejection": False,
    "may_authorize_adaptive_refresh_implementation": True,
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


def _catalog(directory: Path, artifact: str, summary: dict, status: str) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
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
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
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
    _write(CANONICAL_SUMMARY, catalog)


def _parent_summary() -> dict:
    summary = _read(PARENT_ARTIFACT / "summary.json")
    if (
        summary["passed"]
        or not summary["diagnostic_exact_refresh_authorized"]
        or not summary["one_exact_Jacobian_plus_Broyden_policy_rejected"]
    ):
        raise RuntimeError("fixed-Q history rejection changed")
    return summary


def _freeze() -> dict:
    parent = _parent_summary()
    if not _tracked_tree_is_clean():
        raise RuntimeError("diagnostic manifest requires a clean tracked tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "targeted_exact_refresh_diagnostic_manifest_frozen_"
            "execution_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "diagnostic_execution_authorized": True,
        "adaptive_refresh_implementation_authorized": False,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    MANIFEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(MANIFEST_DIRECTORY / "execution_manifest.json", DIAGNOSTIC_CONTRACT)
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
            "parent_summary_sha256": _sha(PARENT_ARTIFACT / "summary.json"),
            "parent_classification": parent["classification"],
        },
    )
    files = ("execution_manifest.json", "provenance.json", "summary.json")
    (MANIFEST_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(MANIFEST_DIRECTORY / name)}  {name}\n" for name in files
        ),
        encoding="utf-8",
    )
    _catalog(MANIFEST_DIRECTORY, MANIFEST_ARTIFACT, summary, "PROSPECTIVE")
    return summary


def _execute() -> dict:
    parent = _parent_summary()
    manifest_summary = _read(MANIFEST_DIRECTORY / "summary.json")
    if not manifest_summary["diagnostic_execution_authorized"]:
        raise RuntimeError("exact-refresh diagnostic is not authorized")
    if not _tracked_tree_is_clean():
        raise RuntimeError("exact-refresh diagnostic requires a clean tree")
    source_path = PARENT_ARTIFACT / "decisive_arrays.npz"
    with np.load(source_path, allow_pickle=False) as source:
        source_state = np.asarray(source["primitive_charts"], dtype=float)
        source_multiplier = np.asarray(source["multipliers"], dtype=float)
    data = f24e1._state_data("primary_20ms")
    initial_increment = (
        (source_state - data["state"]) / data["columns"]
    ).ravel()
    timestep = float(DIAGNOSTIC_CONTRACT["timestep_seconds"])
    top_left = (
        data["reaction"].descriptor_scaled_matrix / timestep
        + data["tangent"].evolving_scaled_jacobian
    )
    began = time.perf_counter()
    result = solve_causal_five_field_fixed_q_backward_euler(
        data["context"],
        data["state"],
        timestep,
        data["continuous_rate"],
        source_multiplier,
        top_left,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        q3_target=data["reaction"].q3_value,
        constraint_row_scales=data["reaction"].q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=data["reaction"].raw_schur_inverse,
        residual_tolerance=DIAGNOSTIC_CONTRACT["maximum_scaled_residual"],
        constraint_tolerance=(
            DIAGNOSTIC_CONTRACT["maximum_Q3_relative_defect"]
        ),
        ledger_tolerance=DIAGNOSTIC_CONTRACT["maximum_ledger_relative_defect"],
        storage_parity_tolerance=(
            DIAGNOSTIC_CONTRACT["maximum_storage_parity_relative_defect"]
        ),
        minimum_reconstruction_factor=(
            DIAGNOSTIC_CONTRACT["minimum_path_reconstruction_factor"]
        ),
        maximum_schur_condition_number=(
            DIAGNOSTIC_CONTRACT["maximum_raw_Schur_condition_number"]
        ),
        maximum_scaled_primitive_change=(
            DIAGNOSTIC_CONTRACT["maximum_scaled_primitive_change"]
        ),
        maximum_newton_iterations=(
            DIAGNOSTIC_CONTRACT["maximum_newton_iterations"]
        ),
        maximum_line_search_iterations=(
            DIAGNOSTIC_CONTRACT["maximum_line_search_iterations"]
        ),
        refresh_exact_jacobian=True,
        maximum_exact_jacobian_refreshes=1,
        initial_scaled_increment=initial_increment,
        base_reaction=data["reaction"],
        physical_state_audit=_state_audit,
        require_physical_state_audit=True,
        maximum_h_over_r=DIAGNOSTIC_CONTRACT["maximum_H_over_R"],
        minimum_scattering_optical_depth=(
            DIAGNOSTIC_CONTRACT["minimum_scattering_optical_depth"]
        ),
        progress_callback=lambda payload: print(
            f"f24e2 exact refresh: {payload}", flush=True
        ),
    )
    metrics = f24e1._result_metrics(result, data)
    metrics["wall_seconds"] = time.perf_counter() - began
    nonroot_passed = all(
        value
        for key, value in metrics["acceptance"].items()
        if key
        not in {
            "accepted",
            "nonlinear_root_passed",
            "complete_residual_passed",
            "failure_reasons",
        }
    )
    root_reached = bool(
        result.maximum_scaled_residual
        <= DIAGNOSTIC_CONTRACT["maximum_scaled_residual"]
    )
    diagnostic_passed = bool(root_reached and result.accepted)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "targeted_exact_refresh_reached_root_"
            "adaptive_refresh_implementation_authorized"
            if diagnostic_passed
            else "targeted_exact_refresh_did_not_reach_root_"
            "endpoint_linearization_audit_authorized"
        ),
        "passed": diagnostic_passed,
        "diagnostic_only": True,
        "parent_rejection_preserved": True,
        "root_reached": root_reached,
        "all_nonroot_gates_passed": nonroot_passed,
        "physical_failure_detected": not nonroot_passed,
        "adaptive_refresh_implementation_authorized": diagnostic_passed,
        "endpoint_linearization_audit_authorized": not diagnostic_passed,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(RESULT_DIRECTORY / "summary.json", summary)
    _write(RESULT_DIRECTORY / "metrics.json", metrics)
    _write(RESULT_DIRECTORY / "contract.json", DIAGNOSTIC_CONTRACT)
    np.savez_compressed(
        RESULT_DIRECTORY / "decisive_arrays.npz",
        primitive_charts=result.primitive_charts,
        primitive_increment=result.primitive_increment,
        scaled_rate_per_s=result.scaled_rate_per_s,
        multipliers=result.multipliers,
        scaled_reaction_rate_action_per_s=(
            result.scaled_reaction_rate_action_per_s
        ),
        augmented_scaled_residual=(
            result.evaluation.augmented_scaled_residual
        ),
    )
    _write(
        RESULT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "untracked_files_at_start": _git(
                "ls-files", "--others", "--exclude-standard"
            ).splitlines(),
            "runner_sha256": _sha(ROOT / THIS_RUNNER),
            "manifest_summary_sha256": _sha(
                MANIFEST_DIRECTORY / "summary.json"
            ),
            "parent_summary_sha256": _sha(PARENT_ARTIFACT / "summary.json"),
            "parent_decisive_arrays_sha256": _sha(source_path),
            "parent_classification": parent["classification"],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "blas_thread_environment": {
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
        "contract.json",
        "decisive_arrays.npz",
        "metrics.json",
        "provenance.json",
        "summary.json",
    )
    (RESULT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(RESULT_DIRECTORY / name)}  {name}\n" for name in files
        ),
        encoding="utf-8",
    )
    _catalog(
        RESULT_DIRECTORY,
        RESULT_ARTIFACT,
        summary,
        "DIAGNOSTIC" if diagnostic_passed else "REJECTED",
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
