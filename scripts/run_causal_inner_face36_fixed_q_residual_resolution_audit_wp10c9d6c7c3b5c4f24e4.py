#!/usr/bin/env python3
"""Freeze and run the blockwise fixed-Q residual-resolution audit."""

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

import run_causal_inner_face36_state_dependent_fixed_q_step_preflight_wp10c9d6c7c3b5c4f24 as f24  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_fixed_q_reaction,
    evaluate_causal_five_field_fixed_q_bdf,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e4"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_residual_resolution_audit_manifest_"
    "wp10c9d6c7c3b5c4f24e4"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_residual_resolution_audit_"
    "wp10c9d6c7c3b5c4f24e4"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / RESULT_ARTIFACT
PARENT_ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_endpoint_linearization_audit_"
    "wp10c9d6c7c3b5c4f24e3"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_residual_resolution_audit_"
    "wp10c9d6c7c3b5c4f24e4.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_residual_resolution_audit_"
    "wp10c9d6c7c3b5c4f24e4.py"
)
ALPHAS = tuple(2.0 ** (-index) for index in range(8))
CONTRACT = {
    "schema_version": 1,
    "analysis_only": True,
    "trajectory_executed": False,
    "physical_operator_changed": False,
    "source": "f24e3_saved_endpoint_and_exact_Newton_correction",
    "timestep_seconds": 1.0e-7,
    "line_search_alphas": list(ALPHAS),
    "endpoint_repeat_evaluations": 3,
    "outer_localization_cells": [108, 109, 110, 111],
    "minimum_resolvable_model_error_order": 1.5,
    "maximum_full_step_model_error_to_base_residual": 0.10,
    "maximum_increment_direct_storage_relative_defect": 1.0e-9,
    "require_bitwise_endpoint_repeatability": True,
    "may_change_physical_equations": False,
    "may_change_row_scales_or_merit_norm": False,
    "may_relax_nonlinear_residual_gate": False,
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


def _relative(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


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
    if summary["passed"] or not summary["endpoint_derivative_precision_repair_authorized"]:
        raise RuntimeError("endpoint linearization classification changed")
    return summary


def _freeze() -> dict:
    _parent_summary()
    if not _tracked_tree_is_clean():
        raise RuntimeError("residual-resolution manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_residual_resolution_manifest_frozen_analysis_authorized"
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
            "parent_summary_sha256": _sha(PARENT_ARTIFACT / "summary.json"),
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


def _blocks(evaluation, rows: np.ndarray) -> dict[str, np.ndarray]:
    monolithic = evaluation.monolithic_evaluation
    row_values = np.asarray(rows, dtype=float).ravel()
    mapped = np.asarray(monolithic.mapped_temporal_storage_rows).ravel() / row_values
    height = (
        np.asarray(monolithic.responsive_height_temporal_storage_rows).ravel()
        / row_values
    )
    stationary = evaluation.scaled_monolithic_residual - mapped - height
    return {
        "mapped_storage": mapped,
        "height_storage": height,
        "stationary": stationary,
        "reaction": np.asarray(evaluation.scaled_reaction_residual),
        "constraint": np.asarray(evaluation.scaled_constraint_residual),
        "augmented": np.asarray(evaluation.augmented_scaled_residual),
    }


def _data() -> dict:
    (
        layout,
        configuration,
        _trajectory,
        _index,
        _old,
        state,
        _timestep,
        _previous_timestep,
        _history,
    ) = f24._endpoint_data()
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(state.shape)
    rows = np.asarray(configuration["rows"], dtype=float).reshape(state.shape)
    reaction = causal_five_field_fixed_q_reaction(
        context,
        state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
        maximum_schur_condition_number=1.0e8,
    )
    return {
        "layout": layout,
        "context": context,
        "state": np.asarray(state, dtype=float),
        "columns": columns,
        "rows": rows,
        "reaction": reaction,
    }


def _evaluate_unknown(
    unknown: np.ndarray,
    data: dict,
    *,
    direct_rate: bool = False,
):
    dimensions = int(data["state"].size)
    candidate = data["state"] + data["columns"] * unknown[:dimensions].reshape(
        data["state"].shape
    )
    return evaluate_causal_five_field_fixed_q_bdf(
        data["state"],
        candidate,
        unknown[dimensions:],
        data["reaction"].q3_value,
        CONTRACT["timestep_seconds"],
        data["context"],
        order=1,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        constraint_row_scales=data["reaction"].q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=data["reaction"].raw_schur_inverse,
        scaled_rate_per_s=(
            unknown[:dimensions] / CONTRACT["timestep_seconds"]
            if direct_rate
            else None
        ),
        maximum_schur_condition_number=1.0e8,
    )


def _alpha_path(alpha: float) -> Path:
    token = f"{alpha:.12e}".replace("+", "p").replace("-", "m").replace(".", "d")
    return CHECKPOINT_DIRECTORY / f"alpha_{token}.npz"


def _evaluate_alpha(alpha: float, unknown: np.ndarray, correction: np.ndarray, data: dict):
    path = _alpha_path(alpha)
    if path.exists():
        with np.load(path, allow_pickle=False) as source:
            return {name: np.asarray(source[name]) for name in source.files}
    print(f"f24e4: evaluate alpha={alpha:.8f}", flush=True)
    blocks = _blocks(
        _evaluate_unknown(unknown + alpha * correction, data),
        data["rows"],
    )
    _write_npz(path, **blocks)
    return blocks


def _order(coarse: float, fine: float) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return float(math.log(coarse / fine) / math.log(2.0))


def _execute() -> dict:
    _parent_summary()
    manifest = _read(MANIFEST_DIRECTORY / "summary.json")
    if not manifest["analysis_execution_authorized"]:
        raise RuntimeError("residual-resolution audit is not authorized")
    if not _tracked_tree_is_clean():
        raise RuntimeError("residual-resolution audit requires a clean tree")
    identity = {
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "manifest_summary_sha256": _sha(MANIFEST_DIRECTORY / "summary.json"),
        "parent_arrays_sha256": _sha(PARENT_ARTIFACT / "decisive_arrays.npz"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    identity_path = CHECKPOINT_DIRECTORY / "execution_identity.json"
    if identity_path.exists() and _read(identity_path) != _plain(identity):
        raise RuntimeError("residual-resolution execution identity changed")
    _write(identity_path, identity)
    began = time.perf_counter()
    data = _data()
    with np.load(PARENT_ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as source:
        unknown = np.asarray(source["unknown"], dtype=float)
        correction = np.asarray(source["correction"], dtype=float)
        direction = np.asarray(source["direction"], dtype=float)
        analytic_directional = {
            "mapped_storage": np.asarray(source["mapped_analytic_JVP"]),
            "height_storage": np.asarray(source["height_analytic_JVP"]),
            "stationary": np.asarray(source["stationary_analytic_JVP"]),
            "reaction": np.asarray(source["reaction_analytic_JVP"]),
            "constraint": np.asarray(source["constraint_analytic_JVP"]),
            "augmented": np.asarray(source["exact_matrix_action"]),
        }
    correction_norm = float(np.linalg.norm(correction))
    if not np.allclose(
        direction,
        correction / correction_norm,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("saved endpoint direction no longer closes")
    base_evaluations = [
        _evaluate_unknown(unknown, data)
        for _index in range(int(CONTRACT["endpoint_repeat_evaluations"]))
    ]
    base_blocks = [_blocks(item, data["rows"]) for item in base_evaluations]
    repeat_bitwise = all(
        all(
            np.array_equal(base_blocks[0][name], current[name])
            for name in base_blocks[0]
        )
        for current in base_blocks[1:]
    )
    direct_blocks = _blocks(
        _evaluate_unknown(unknown, data, direct_rate=True),
        data["rows"],
    )
    storage_increment_direct_defects = {
        name: _relative(base_blocks[0][name], direct_blocks[name])
        for name in ("mapped_storage", "height_storage")
    }
    actual = [
        _evaluate_alpha(alpha, unknown, correction, data) for alpha in ALPHAS
    ]
    base = base_blocks[0]
    base_norm = max(
        float(np.linalg.norm(base["augmented"])),
        np.finfo(float).tiny,
    )
    analytic_action = {
        name: correction_norm * values
        for name, values in analytic_directional.items()
    }
    alpha_metrics = []
    model_errors = []
    block_model_errors = {name: [] for name in analytic_action}
    for alpha, blocks in zip(ALPHAS, actual, strict=True):
        predicted = base["augmented"] + alpha * analytic_action["augmented"]
        model_error = float(
            np.linalg.norm(blocks["augmented"] - predicted) / base_norm
        )
        model_errors.append(model_error)
        current_block_errors = {}
        for name in analytic_action:
            actual_change = blocks[name] - base[name]
            predicted_change = alpha * analytic_action[name]
            error = float(
                np.linalg.norm(actual_change - predicted_change) / base_norm
            )
            block_model_errors[name].append(error)
            current_block_errors[name] = error
        alpha_metrics.append(
            {
                "alpha": alpha,
                "augmented_L2_norm": float(np.linalg.norm(blocks["augmented"])),
                "augmented_maximum": float(np.max(np.abs(blocks["augmented"]))),
                "predicted_L2_norm": float(np.linalg.norm(predicted)),
                "model_error_to_base_residual": model_error,
                "block_model_errors_to_base_residual": current_block_errors,
            }
        )
    model_orders = [
        _order(model_errors[index], model_errors[index + 1])
        for index in range(len(model_errors) - 1)
    ]
    outer = np.asarray(CONTRACT["outer_localization_cells"], dtype=int)
    outer_rows = np.concatenate(
        [np.arange(5 * cell, 5 * cell + 5, dtype=int) for cell in outer]
    )
    outer_metrics = {}
    for name in ("mapped_storage", "height_storage", "stationary", "reaction"):
        errors = []
        for alpha, blocks in zip(ALPHAS, actual, strict=True):
            actual_change = blocks[name][outer_rows] - base[name][outer_rows]
            predicted_change = alpha * analytic_action[name][outer_rows]
            errors.append(
                float(np.linalg.norm(actual_change - predicted_change) / base_norm)
            )
        outer_metrics[name] = errors
    block_norm_sum = sum(
        float(np.linalg.norm(base[name]))
        for name in ("mapped_storage", "height_storage", "stationary", "reaction")
    )
    cancellation_ratio = float(base_norm / max(block_norm_sum, np.finfo(float).tiny))
    repeat_passed = bool(
        repeat_bitwise or not CONTRACT["require_bitwise_endpoint_repeatability"]
    )
    storage_passed = bool(
        max(storage_increment_direct_defects.values())
        <= CONTRACT["maximum_increment_direct_storage_relative_defect"]
    )
    full_step_passed = bool(
        model_errors[0]
        <= CONTRACT["maximum_full_step_model_error_to_base_residual"]
    )
    quadratic_passed = bool(
        min(model_orders[:3])
        >= CONTRACT["minimum_resolvable_model_error_order"]
    )
    passed = bool(
        repeat_passed and storage_passed and full_step_passed and quadratic_passed
    )
    if not repeat_passed:
        classification = "fixed_Q_endpoint_residual_not_repeatable"
        selected = "residual_repeatability_repair"
    elif not storage_passed:
        classification = "fixed_Q_endpoint_storage_representation_mismatch"
        selected = "mapped_or_height_storage_repair"
    elif not full_step_passed or not quadratic_passed:
        classification = (
            "fixed_Q_endpoint_residual_linearization_floor_"
            "block_localization_authorized"
        )
        selected = "cancellation_prone_block_repair"
    else:
        classification = (
            "fixed_Q_endpoint_residual_resolution_certified_"
            "merit_scaling_audit_authorized"
        )
        selected = "scale_invariant_merit_audit"
    metrics = {
        "endpoint_repeat_bitwise": repeat_bitwise,
        "storage_increment_direct_relative_defects": (
            storage_increment_direct_defects
        ),
        "base_residual_L2_norm": base_norm,
        "base_block_L2_norm_sum": block_norm_sum,
        "base_cancellation_ratio": cancellation_ratio,
        "correction_L2_norm": correction_norm,
        "alpha_metrics": alpha_metrics,
        "model_error_orders": model_orders,
        "block_model_errors_to_base_residual": block_model_errors,
        "outer_cell_model_errors_to_base_residual": outer_metrics,
        "repeatability_passed": repeat_passed,
        "storage_parity_passed": storage_passed,
        "full_step_model_passed": full_step_passed,
        "quadratic_model_passed": quadratic_passed,
        "selected_next": selected,
        "wall_seconds": time.perf_counter() - began,
    }
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "analysis_only": True,
        "trajectory_executed": False,
        "physical_failure_detected": False,
        "parent_rejections_preserved": True,
        "selected_next": selected,
        "physical_history_ladder_authorized": False,
        "adaptive_refresh_implementation_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(RESULT_DIRECTORY / "summary.json", summary)
    _write(RESULT_DIRECTORY / "metrics.json", metrics)
    _write(RESULT_DIRECTORY / "contract.json", CONTRACT)
    arrays = {
        "unknown": unknown,
        "correction": correction,
        "alphas": np.asarray(ALPHAS),
        "model_errors_to_base_residual": np.asarray(model_errors),
        "model_error_orders": np.asarray(model_orders),
        "base_augmented_residual": base["augmented"],
    }
    for index, blocks in enumerate(actual):
        for name, values in blocks.items():
            arrays[f"alpha_{index}_{name}"] = values
    for name, values in analytic_action.items():
        arrays[f"actual_correction_{name}_action"] = values
    _write_npz(RESULT_DIRECTORY / "decisive_arrays.npz", **arrays)
    _write(
        RESULT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            **identity,
            "tracked_worktree_clean_at_start": True,
            "untracked_files_at_start": _git(
                "ls-files", "--others", "--exclude-standard"
            ).splitlines(),
            "parent_summary_sha256": _sha(PARENT_ARTIFACT / "summary.json"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
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
        "DIAGNOSTIC" if passed else "REJECTED",
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
