#!/usr/bin/env python3
"""Freeze and execute one exact correction at the rejected held-out BDF2 endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1 as e1  # noqa: E402
import run_causal_inner_face36_fixed_q_exact_refresh_diagnostic_wp10c9d6c7c3b5c4f24e2 as e2  # noqa: E402
from run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a import (  # noqa: E402
    _state_audit,
)

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_fixed_q_reaction,
    load_causal_five_field_fixed_q_bdf_restart,
    solve_causal_five_field_fixed_q_bdf,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e10"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_heldout_bdf2_exact_refresh_manifest_"
    "wp10c9d6c7c3b5c4f24e10"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_heldout_bdf2_exact_refresh_"
    "wp10c9d6c7c3b5c4f24e10"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
PARENT_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_remaining_history_ladder_stage_"
    "heldout_coarse_wp10c9d6c7c3b5c4f24e9"
)
SOURCE_RESTART = ROOT / "outputs/checkpoints" / (
    "causal_inner_face36_fixed_q_remaining_history_ladder_"
    "wp10c9d6c7c3b5c4f24e9/heldout_coarse_restart.npz"
)
CANONICAL_RESTART = MANIFEST_DIRECTORY / "source_restart.npz"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_heldout_bdf2_exact_refresh_"
    "wp10c9d6c7c3b5c4f24e10.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_heldout_bdf2_exact_refresh_"
    "wp10c9d6c7c3b5c4f24e10.py"
)
CONTRACT = {
    "schema_version": 1,
    "source_case": "heldout_coarse",
    "source_state": "heldout_16ms",
    "source_stage": "rejected_BDF2_endpoint",
    "timestep_seconds": 1.0e-7,
    "expected_source_maximum_scaled_residual": 1.562552753853197e-9,
    "binding_temporal_form": "exact_increment_primary_BDF2",
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
    "maximum_newton_iterations": 1,
    "maximum_line_search_iterations": 12,
    "exact_Jacobian_corrections": 1,
    "diagnostic_only": True,
    "may_amend_parent_rejection": False,
    "may_resume_remaining_ladder": False,
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


def _parent() -> tuple[dict, dict]:
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    if (
        summary["passed"]
        or summary["classification"]
        != "fixed_Q_remaining_history_stage_heldout_coarse_failed"
        or metrics["failed_stage"] != "BDF2"
        or metrics["BDF2"]["acceptance"]["accepted"]
    ):
        raise RuntimeError("held-out coarse BDF2 rejection changed")
    observed = metrics["BDF2"]["maximum_scaled_residual"]
    if observed != CONTRACT["expected_source_maximum_scaled_residual"]:
        raise RuntimeError("held-out coarse rejected endpoint changed")
    return summary, metrics


def _freeze() -> dict:
    parent_summary, _ = _parent()
    if not SOURCE_RESTART.exists():
        raise RuntimeError("held-out coarse accepted BDF1 restart is missing")
    if not _tracked_tree_is_clean():
        raise RuntimeError("exact-refresh manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "heldout_BDF2_exact_refresh_manifest_frozen_execution_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "diagnostic_execution_authorized": True,
        "remaining_ladder_execution_authorized": False,
        "adaptive_refresh_policy_manifest_authorized": False,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    MANIFEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_RESTART, CANONICAL_RESTART)
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
            "parent_summary_sha256": _sha(PARENT_DIRECTORY / "summary.json"),
            "parent_metrics_sha256": _sha(PARENT_DIRECTORY / "metrics.json"),
            "parent_decisive_arrays_sha256": _sha(
                PARENT_DIRECTORY / "decisive_arrays.npz"
            ),
            "source_restart_sha256": _sha(SOURCE_RESTART),
            "canonical_restart_sha256": _sha(CANONICAL_RESTART),
            "parent_classification": parent_summary["classification"],
        },
    )
    names = (
        "execution_manifest.json",
        "provenance.json",
        "source_restart.npz",
        "summary.json",
    )
    (MANIFEST_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(MANIFEST_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    e2._catalog(MANIFEST_DIRECTORY, MANIFEST_ARTIFACT, summary, "PROSPECTIVE")
    return summary


def _execute() -> dict:
    parent_summary, parent_metrics = _parent()
    manifest = _read(MANIFEST_DIRECTORY / "summary.json")
    if not manifest["diagnostic_execution_authorized"]:
        raise RuntimeError("held-out BDF2 exact refresh is not authorized")
    if not _tracked_tree_is_clean():
        raise RuntimeError("held-out BDF2 exact refresh requires a clean tree")
    restart_hash = _sha(CANONICAL_RESTART)
    if restart_hash != _read(MANIFEST_DIRECTORY / "provenance.json")[
        "canonical_restart_sha256"
    ]:
        raise RuntimeError("canonical BDF1 restart changed")

    data = e1._state_data("heldout_16ms")
    restart = load_causal_five_field_fixed_q_bdf_restart(
        CANONICAL_RESTART,
        data["context"],
    )
    source_path = PARENT_DIRECTORY / "decisive_arrays.npz"
    with np.load(source_path, allow_pickle=False) as source:
        source_state = np.asarray(source["bdf2_primitive_charts"], dtype=float)
        source_increment = np.asarray(
            source["bdf2_primitive_increment"], dtype=float
        )
        source_rate = np.asarray(source["bdf2_scaled_rate_per_s"], dtype=float)
        source_multiplier = np.asarray(source["bdf2_multipliers"], dtype=float)
        source_residual = np.asarray(
            source["bdf2_augmented_scaled_residual"], dtype=float
        )
    if float(np.max(np.abs(source_residual))) != CONTRACT[
        "expected_source_maximum_scaled_residual"
    ]:
        raise RuntimeError("canonical rejected BDF2 residual changed")
    reconstructed_state = restart.primitive_charts + source_increment
    if not np.array_equal(reconstructed_state, source_state):
        raise RuntimeError("rejected BDF2 state/increment identity failed")

    timestep = float(CONTRACT["timestep_seconds"])
    endpoint_reaction = causal_five_field_fixed_q_reaction(
        data["context"],
        restart.primitive_charts,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        maximum_schur_condition_number=(
            CONTRACT["maximum_raw_Schur_condition_number"]
        ),
    )
    top_left = (
        1.5 * endpoint_reaction.descriptor_scaled_matrix / timestep
        + data["tangent"].evolving_scaled_jacobian
    )
    progress = []

    def record(payload: dict) -> None:
        progress.append(_plain(payload))
        print(f"f24e10 heldout BDF2 exact refresh: {payload}", flush=True)

    began = time.perf_counter()
    result = solve_causal_five_field_fixed_q_bdf(
        data["context"],
        restart.primitive_charts,
        timestep,
        source_rate,
        source_multiplier,
        top_left,
        order=2,
        history=restart.history,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        q3_target=restart.q3_target,
        constraint_row_scales=restart.constraint_row_scales,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=restart.reaction_channel_transform,
        residual_tolerance=CONTRACT["maximum_scaled_residual"],
        constraint_tolerance=CONTRACT["maximum_Q3_relative_defect"],
        ledger_tolerance=CONTRACT["maximum_ledger_relative_defect"],
        storage_parity_tolerance=(
            CONTRACT["maximum_storage_parity_relative_defect"]
        ),
        minimum_reconstruction_factor=(
            CONTRACT["minimum_path_reconstruction_factor"]
        ),
        maximum_schur_condition_number=(
            CONTRACT["maximum_raw_Schur_condition_number"]
        ),
        maximum_scaled_primitive_change=(
            CONTRACT["maximum_scaled_primitive_change"]
        ),
        maximum_newton_iterations=CONTRACT["maximum_newton_iterations"],
        maximum_line_search_iterations=(
            CONTRACT["maximum_line_search_iterations"]
        ),
        refresh_exact_jacobian=True,
        maximum_exact_jacobian_refreshes=1,
        initial_scaled_increment=(source_increment / data["columns"]).ravel(),
        base_reaction=endpoint_reaction,
        physical_state_audit=_state_audit,
        require_physical_state_audit=True,
        maximum_h_over_r=CONTRACT["maximum_H_over_R"],
        minimum_scattering_optical_depth=(
            CONTRACT["minimum_scattering_optical_depth"]
        ),
        progress_callback=record,
    )
    wall_seconds = time.perf_counter() - began
    if not progress or progress[0]["stage"] != "initial_residual":
        raise RuntimeError("fresh diagnostic did not record its initial residual")
    initial_residual = progress[0]["maximum_scaled_residual"]
    initial_reproduced = bool(
        initial_residual
        == CONTRACT["expected_source_maximum_scaled_residual"]
    )

    metrics = e1._result_metrics(result, data)
    monolithic = result.evaluation.monolithic_evaluation
    metrics.update(
        {
            "wall_seconds": wall_seconds,
            "source_maximum_scaled_residual": float(
                np.max(np.abs(source_residual))
            ),
            "fresh_initial_maximum_scaled_residual": initial_residual,
            "source_endpoint_reproduced_bitwise": initial_reproduced,
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
            "progress": progress,
        }
    )
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
    exact_budget_passed = bool(result.exact_jacobian_assemblies == 1)
    root_reached = bool(
        result.maximum_scaled_residual <= CONTRACT["maximum_scaled_residual"]
    )
    diagnostic_passed = bool(
        initial_reproduced
        and exact_budget_passed
        and root_reached
        and result.accepted
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "heldout_BDF2_exact_refresh_reached_root_"
            "adaptive_policy_manifest_authorized"
            if diagnostic_passed
            else "heldout_BDF2_exact_refresh_did_not_reach_root_"
            "endpoint_linearization_audit_authorized"
        ),
        "passed": diagnostic_passed,
        "diagnostic_only": True,
        "parent_rejection_preserved": True,
        "source_endpoint_reproduced_bitwise": initial_reproduced,
        "one_exact_Jacobian_correction_used": exact_budget_passed,
        "root_reached": root_reached,
        "all_nonroot_gates_passed": nonroot_passed,
        "physical_failure_detected": not nonroot_passed,
        "adaptive_refresh_policy_manifest_authorized": diagnostic_passed,
        "endpoint_linearization_audit_authorized": not diagnostic_passed,
        "remaining_ladder_execution_authorized": False,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(RESULT_DIRECTORY / "summary.json", summary)
    _write(RESULT_DIRECTORY / "metrics.json", metrics)
    _write(RESULT_DIRECTORY / "contract.json", CONTRACT)
    np.savez_compressed(
        RESULT_DIRECTORY / "decisive_arrays.npz",
        source_primitive_charts=source_state,
        source_primitive_increment=source_increment,
        source_multipliers=source_multiplier,
        source_augmented_scaled_residual=source_residual,
        corrected_primitive_charts=result.primitive_charts,
        corrected_primitive_increment=result.primitive_increment,
        corrected_scaled_rate_per_s=result.scaled_rate_per_s,
        corrected_scaled_interval_rate_per_s=(
            result.scaled_interval_rate_per_s
        ),
        corrected_multipliers=result.multipliers,
        corrected_scaled_reaction_rate_action_per_s=(
            result.scaled_reaction_rate_action_per_s
        ),
        corrected_augmented_scaled_residual=(
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
            "runner_sha256": _sha(ROOT / THIS_RUNNER),
            "test_sha256": _sha(ROOT / THIS_TEST),
            "manifest_summary_sha256": _sha(
                MANIFEST_DIRECTORY / "summary.json"
            ),
            "canonical_restart_sha256": restart_hash,
            "parent_summary_sha256": _sha(PARENT_DIRECTORY / "summary.json"),
            "parent_metrics_sha256": _sha(PARENT_DIRECTORY / "metrics.json"),
            "parent_decisive_arrays_sha256": _sha(source_path),
            "parent_classification": parent_summary["classification"],
            "parent_failed_residual": parent_metrics["BDF2"][
                "maximum_scaled_residual"
            ],
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
