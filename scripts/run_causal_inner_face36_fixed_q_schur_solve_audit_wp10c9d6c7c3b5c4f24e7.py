#!/usr/bin/env python3
"""Freeze and run the fixed-Q three-channel Schur solve audit."""

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

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_fixed_q_reaction,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e7"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_schur_solve_audit_manifest_"
    "wp10c9d6c7c3b5c4f24e7"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_schur_solve_audit_"
    "wp10c9d6c7c3b5c4f24e7"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
PARENT_DIRECTORY = e6.RESULT_DIRECTORY
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_schur_solve_audit_"
    "wp10c9d6c7c3b5c4f24e7.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_schur_solve_audit_"
    "wp10c9d6c7c3b5c4f24e7.py"
)
CONTRACT = {
    "schema_version": 1,
    "analysis_only": True,
    "states": ["committed_middle_20ms", "recovered_BDF1_endpoint"],
    "methods": [
        "direct_LU",
        "global_scaled_LU",
        "row_scaled_LU",
        "row_column_equilibrated_LU",
        "global_scaled_QR",
        "global_scaled_SVD",
        "row_column_equilibrated_LU_refined_1",
        "row_column_equilibrated_LU_refined_2",
    ],
    "maximum_selected_solve_closure_defect": 5.0e-13,
    "maximum_selected_physical_action_relative_difference": 1.0e-10,
    "maximum_condition_number": 1.0e8,
    "required_rank": 3,
    "selection": "minimum_double_precision_identity_closure",
    "may_change_reaction_support_or_physical_rows": False,
    "may_relax_ledger_or_condition_gates": False,
    "physical_execution_authorized": False,
    "fixed_Q_micro_solver_authorized": False,
    "reduced_slow_evolution_authorized": False,
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _parent_summary() -> dict:
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    solve_defect = metrics["case"]["BDF1"][
        "maximum_raw_Schur_solve_relative_defect"
    ]
    if (
        summary["passed"]
        or summary["classification"] != "fixed_Q_primary_case_recovery_failed"
        or metrics["case"]["BDF1"]["failure_reasons"]
        != ["reaction_conditioning"]
        or not 1.0e-12 < solve_defect < 1.1e-12
    ):
        raise RuntimeError("primary recovery classification changed")
    return summary


def _freeze() -> dict:
    _parent_summary()
    if not _tracked_tree_is_clean():
        raise RuntimeError("Schur manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_schur_solve_audit_manifest_frozen_analysis_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "analysis_execution_authorized": True,
        "physical_execution_authorized": False,
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
    e6.WORK_PACKAGE = WORK_PACKAGE
    e6._catalog(
        MANIFEST_DIRECTORY,
        MANIFEST_ARTIFACT,
        summary,
        "PROSPECTIVE",
    )
    return summary


def _closure(matrix: np.ndarray, inverse: np.ndarray) -> tuple[float, float]:
    identity = np.eye(3)
    double = float(np.linalg.norm(matrix @ inverse - identity) / np.linalg.norm(identity))
    extended_residual = (
        matrix.astype(np.longdouble) @ inverse.astype(np.longdouble)
        - identity.astype(np.longdouble)
    )
    extended = float(
        np.sqrt(np.sum(extended_residual * extended_residual))
        / np.sqrt(np.longdouble(3.0))
    )
    return double, extended


def _refine(
    matrix: np.ndarray,
    inverse: np.ndarray,
    iterations: int,
) -> np.ndarray:
    scale = float(np.max(np.abs(matrix)))
    normalized = matrix / scale
    current = np.array(inverse, copy=True)
    identity = np.eye(3, dtype=np.longdouble)
    extended_matrix = matrix.astype(np.longdouble)
    for _index in range(iterations):
        residual = identity - extended_matrix @ current.astype(np.longdouble)
        correction = np.linalg.solve(normalized, np.asarray(residual, dtype=float))
        current += correction / scale
    return current


def _candidates(matrix: np.ndarray) -> dict[str, np.ndarray]:
    identity = np.eye(3)
    scale = float(np.max(np.abs(matrix)))
    normalized = matrix / scale
    direct = np.linalg.solve(matrix, identity)
    global_scaled = np.linalg.solve(normalized, identity) / scale
    row_scale = np.max(np.abs(matrix), axis=1)
    row_scaled = matrix / row_scale[:, None]
    row_inverse = np.linalg.solve(row_scaled, np.diag(1.0 / row_scale))
    column_scale = np.max(np.abs(row_scaled), axis=0)
    equilibrated = row_scaled / column_scale[None, :]
    equilibrated_inverse = (
        np.diag(1.0 / column_scale)
        @ np.linalg.solve(equilibrated, np.diag(1.0 / row_scale))
    )
    q, r = np.linalg.qr(normalized)
    qr_inverse = np.linalg.solve(r, q.T) / scale
    u, singular, vt = np.linalg.svd(normalized)
    svd_inverse = (vt.T * (1.0 / singular)) @ u.T / scale
    return {
        "direct_LU": direct,
        "global_scaled_LU": global_scaled,
        "row_scaled_LU": row_inverse,
        "row_column_equilibrated_LU": equilibrated_inverse,
        "global_scaled_QR": qr_inverse,
        "global_scaled_SVD": svd_inverse,
        "row_column_equilibrated_LU_refined_1": _refine(
            matrix,
            equilibrated_inverse,
            1,
        ),
        "row_column_equilibrated_LU_refined_2": _refine(
            matrix,
            equilibrated_inverse,
            2,
        ),
    }


def _state_metrics(reaction, multiplier: np.ndarray) -> tuple[dict, dict]:
    matrix = np.asarray(reaction.raw_schur_matrix, dtype=float)
    candidates = _candidates(matrix)
    baseline_action = (
        reaction.raw_reaction_lift
        @ reaction.raw_schur_inverse
        @ multiplier
    )
    metrics = {}
    for name, inverse in candidates.items():
        double, extended = _closure(matrix, inverse)
        action = reaction.raw_reaction_lift @ inverse @ multiplier
        action_scale = max(
            float(np.linalg.norm(action)),
            float(np.linalg.norm(baseline_action)),
            np.finfo(float).tiny,
        )
        metrics[name] = {
            "double_identity_closure_defect": double,
            "extended_identity_closure_defect": extended,
            "physical_action_relative_difference_from_direct": float(
                np.linalg.norm(action - baseline_action) / action_scale
            ),
        }
    selected = min(
        candidates,
        key=lambda name: metrics[name]["double_identity_closure_defect"],
    )
    return (
        {
            "raw_schur_matrix": matrix,
            "raw_schur_singular_values": reaction.raw_schur_singular_values,
            "raw_schur_rank": reaction.raw_schur_numerical_rank,
            "raw_schur_condition_number": reaction.raw_schur_condition_number,
            "methods": metrics,
            "selected_method": selected,
        },
        candidates,
    )


def _execute() -> dict:
    _parent_summary()
    manifest = _read(MANIFEST_DIRECTORY / "summary.json")
    if not manifest["analysis_execution_authorized"]:
        raise RuntimeError("Schur audit is not authorized")
    if not _tracked_tree_is_clean():
        raise RuntimeError("Schur audit requires a clean tree")
    data = e1._state_data("primary_20ms")
    with np.load(PARENT_DIRECTORY / "decisive_arrays.npz", allow_pickle=False) as source:
        endpoint = np.asarray(source["bdf1_primitive_charts"], dtype=float)
        endpoint_multiplier = np.asarray(source["bdf1_multipliers"], dtype=float)
    start_multiplier = np.asarray(data["continuous_multiplier"], dtype=float)
    endpoint_reaction = causal_five_field_fixed_q_reaction(
        data["context"],
        endpoint,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        maximum_schur_condition_number=CONTRACT["maximum_condition_number"],
    )
    state_metrics = {}
    arrays = {}
    for label, reaction, multiplier in (
        ("start", data["reaction"], start_multiplier),
        ("endpoint", endpoint_reaction, endpoint_multiplier),
    ):
        current, candidates = _state_metrics(reaction, multiplier)
        state_metrics[label] = current
        arrays[f"{label}_raw_schur_matrix"] = reaction.raw_schur_matrix
        for name, inverse in candidates.items():
            arrays[f"{label}_{name}_inverse"] = inverse
    selected_methods = {
        label: values["selected_method"] for label, values in state_metrics.items()
    }
    common_method = (
        selected_methods["start"]
        if selected_methods["start"] == selected_methods["endpoint"]
        else None
    )
    selected_metrics = (
        []
        if common_method is None
        else [
            values["methods"][common_method] for values in state_metrics.values()
        ]
    )
    passed = bool(
        common_method is not None
        and all(values["raw_schur_rank"] == CONTRACT["required_rank"] for values in state_metrics.values())
        and all(
            values["raw_schur_condition_number"]
            <= CONTRACT["maximum_condition_number"]
            for values in state_metrics.values()
        )
        and max(
            item["double_identity_closure_defect"] for item in selected_metrics
        )
        <= CONTRACT["maximum_selected_solve_closure_defect"]
        and max(
            item["physical_action_relative_difference_from_direct"]
            for item in selected_metrics
        )
        <= CONTRACT["maximum_selected_physical_action_relative_difference"]
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_schur_solve_audit_passed_implementation_authorized"
            if passed
            else "fixed_Q_schur_solve_audit_failed"
        ),
        "passed": passed,
        "analysis_only": True,
        "trajectory_executed": False,
        "physical_failure_detected": False,
        "selected_method": common_method,
        "schur_solve_implementation_authorized": passed,
        "physical_execution_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    metrics = {
        "states": state_metrics,
        "selected_methods": selected_methods,
        "common_selected_method": common_method,
    }
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(RESULT_DIRECTORY / "contract.json", CONTRACT)
    _write(RESULT_DIRECTORY / "metrics.json", metrics)
    _write(RESULT_DIRECTORY / "summary.json", summary)
    _write_npz(RESULT_DIRECTORY / "decisive_arrays.npz", **arrays)
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
            "manifest_summary_sha256": _sha(MANIFEST_DIRECTORY / "summary.json"),
            "fixed_q_source_sha256": _sha(
                ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py"
            ),
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
    e6.WORK_PACKAGE = WORK_PACKAGE
    e6._catalog(
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
