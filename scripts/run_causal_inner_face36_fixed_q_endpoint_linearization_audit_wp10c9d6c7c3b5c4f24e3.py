#!/usr/bin/env python3
"""Freeze and run the fail-fast fixed-Q endpoint linearization audit."""

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

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    _equilibrated_dense_solve,
    causal_five_field_fixed_q_augmented_step_matrix,
    evaluate_causal_five_field_fixed_q_bdf,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_discrete_tangent import (  # noqa: E402
    causal_five_field_monolithic_discrete_step_matrix,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e3"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_endpoint_linearization_audit_manifest_"
    "wp10c9d6c7c3b5c4f24e3"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_endpoint_linearization_audit_"
    "wp10c9d6c7c3b5c4f24e3"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / RESULT_ARTIFACT
PARENT_ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_exact_refresh_diagnostic_"
    "wp10c9d6c7c3b5c4f24e2"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_endpoint_linearization_audit_"
    "wp10c9d6c7c3b5c4f24e3.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_endpoint_linearization_audit_"
    "wp10c9d6c7c3b5c4f24e3.py"
)
BINDING_DIRECTIONAL_STEP = 1.0e-4
DIAGNOSTIC_DIRECTIONAL_STEPS = (4.0e-4, 2.0e-4, 5.0e-5, 2.0e-5)
LINE_SEARCH_ALPHAS = tuple(2.0 ** (-index) for index in range(8))
CONTRACT = {
    "schema_version": 1,
    "source": "saved_rejected_primary_coarse_BDF1_endpoint",
    "analysis_only": True,
    "trajectory_executed": False,
    "physical_operator_changed": False,
    "timestep_seconds": 1.0e-7,
    "binding_directional_step": BINDING_DIRECTIONAL_STEP,
    "diagnostic_directional_steps": list(DIAGNOSTIC_DIRECTIONAL_STEPS),
    "line_search_alphas": list(LINE_SEARCH_ALPHAS),
    "maximum_direct_augmented_JVP_relative_defect": 1.0e-8,
    "maximum_block_JVP_relative_defect": 1.0e-8,
    "maximum_central_five_point_relative_defect": 1.0e-8,
    "maximum_actual_correction_action_error_to_residual": 0.10,
    "maximum_equilibrated_linear_relative_residual": 1.0e-10,
    "fail_fast_before_sweep_if_binding_action_fails": True,
    "may_relax_nonlinear_residual_gate": False,
    "may_authorize_physical_execution": False,
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


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(first - second) / scale)


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
    if summary["passed"] or not summary["endpoint_linearization_audit_authorized"]:
        raise RuntimeError("exact-refresh diagnostic classification changed")
    return summary


def _freeze() -> dict:
    _parent_summary()
    if not _tracked_tree_is_clean():
        raise RuntimeError("endpoint audit manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_endpoint_linearization_manifest_frozen_"
            "analysis_authorized"
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


def _evaluate_unknown(unknown: np.ndarray, data: dict):
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
        maximum_schur_condition_number=1.0e8,
    )


def _sample_path(step: float, sign: int) -> Path:
    token = f"{step:.12e}".replace("+", "p").replace("-", "m").replace(".", "d")
    return CHECKPOINT_DIRECTORY / f"directional_{token}_{int(sign):+d}.npz"


def _save_sample(path: Path, blocks: dict[str, np.ndarray]) -> None:
    _write_npz(path, **blocks)


def _load_sample(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _directional_sample(
    step: float,
    sign: int,
    unknown: np.ndarray,
    direction: np.ndarray,
    data: dict,
) -> dict[str, np.ndarray]:
    path = _sample_path(step, sign)
    if path.exists():
        return _load_sample(path)
    print(f"f24e3: evaluate {sign:+d} * {step:.3e}", flush=True)
    evaluation = _evaluate_unknown(unknown + sign * step * direction, data)
    blocks = _blocks(evaluation, data["rows"])
    _save_sample(path, blocks)
    return blocks


def _finite_difference(
    step: float,
    unknown: np.ndarray,
    direction: np.ndarray,
    data: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    plus = _directional_sample(step, 1, unknown, direction, data)
    minus = _directional_sample(step, -1, unknown, direction, data)
    plus_two = _directional_sample(2.0 * step, 1, unknown, direction, data)
    minus_two = _directional_sample(2.0 * step, -1, unknown, direction, data)
    central = {
        name: (plus[name] - minus[name]) / (2.0 * step) for name in plus
    }
    five = {
        name: (
            -plus_two[name]
            + 8.0 * plus[name]
            - 8.0 * minus[name]
            + minus_two[name]
        )
        / (12.0 * step)
        for name in plus
    }
    return central, five


def _component_localization(values: np.ndarray, n_cells: int) -> dict:
    top = np.asarray(values[:-3], dtype=float).reshape(n_cells, 5)
    constraint = np.asarray(values[-3:], dtype=float)
    field_norms = np.linalg.norm(top, axis=0)
    field_maxima = np.max(np.abs(top), axis=0)
    flat_order = np.argsort(np.abs(top).ravel())[::-1][:12]
    largest = [
        {
            "cell": int(index // 5),
            "field": int(index % 5),
            "value": float(top.ravel()[index]),
        }
        for index in flat_order
    ]
    return {
        "field_L2_norms": field_norms,
        "field_maximum_absolute_values": field_maxima,
        "constraint_values": constraint,
        "largest_top_rows": largest,
    }


def _execute() -> dict:
    _parent_summary()
    manifest = _read(MANIFEST_DIRECTORY / "summary.json")
    if not manifest["analysis_execution_authorized"]:
        raise RuntimeError("endpoint linearization audit is not authorized")
    if not _tracked_tree_is_clean():
        raise RuntimeError("endpoint audit requires a clean tracked tree")
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
        raise RuntimeError("endpoint audit execution identity changed")
    _write(identity_path, identity)
    began = time.perf_counter()
    data = f24e1._state_data("primary_20ms")
    dimensions = int(data["state"].size)
    with np.load(PARENT_ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as source:
        endpoint = np.asarray(source["primitive_charts"], dtype=float)
        multipliers = np.asarray(source["multipliers"], dtype=float)
    scaled_increment = ((endpoint - data["state"]) / data["columns"]).ravel()
    unknown = np.concatenate((scaled_increment, multipliers))
    base_evaluation = _evaluate_unknown(unknown, data)
    base_blocks = _blocks(base_evaluation, data["rows"])
    print("f24e3: assemble exact augmented matrix", flush=True)
    matrix = causal_five_field_fixed_q_augmented_step_matrix(
        data["context"],
        data["state"],
        endpoint,
        multipliers,
        CONTRACT["timestep_seconds"],
        None,
        order=1,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        constraint_row_scales=data["reaction"].q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=data["reaction"].raw_schur_inverse,
        reaction=base_evaluation.reaction,
    )
    print("f24e3: assemble monolithic component matrices", flush=True)
    monolithic = causal_five_field_monolithic_discrete_step_matrix(
        data["context"],
        data["state"],
        endpoint,
        CONTRACT["timestep_seconds"],
        None,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        order=1,
    )
    monolithic_parity = _relative(
        matrix.monolithic_scaled_matrix,
        monolithic.scaled_matrix,
    )
    correction, linear_residual = _equilibrated_dense_solve(
        matrix.scaled_matrix,
        -base_blocks["augmented"],
    )
    direct_correction = np.linalg.solve(
        matrix.scaled_matrix,
        -base_blocks["augmented"],
    )
    direct_linear_residual = _relative(
        matrix.scaled_matrix @ direct_correction,
        -base_blocks["augmented"],
    )
    correction_parity = _relative(correction, direct_correction)
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm <= np.finfo(float).tiny:
        raise RuntimeError("endpoint Newton correction is numerically zero")
    direction = correction / correction_norm
    state_direction = direction[:dimensions]
    multiplier_direction = direction[dimensions:]
    coefficient = monolithic.current_increment_coefficient / float(
        monolithic.current_timestep_seconds
    )
    analytic = {
        "mapped_storage": (
            coefficient
            * monolithic.mapped_storage_scaled_matrix
            @ state_direction
        ),
        "height_storage": (
            coefficient
            * monolithic.responsive_height_storage_scaled_matrix
            @ state_direction
        ),
        "stationary": monolithic.stationary_scaled_matrix @ state_direction,
        "reaction": (
            matrix.reaction_state_scaled_matrix @ state_direction
            + matrix.reaction_multiplier_scaled_matrix @ multiplier_direction
        ),
        "constraint": matrix.constraint_scaled_matrix @ state_direction,
        "augmented": matrix.scaled_matrix @ direction,
    }
    central, five = _finite_difference(
        BINDING_DIRECTIONAL_STEP,
        unknown,
        direction,
        data,
    )
    block_relative = {
        name: _relative(analytic[name], five[name]) for name in analytic
    }
    central_five = {
        name: _relative(central[name], five[name]) for name in analytic
    }
    action_error = correction_norm * (
        five["augmented"] - analytic["augmented"]
    )
    residual_norm = max(
        float(np.linalg.norm(base_blocks["augmented"])),
        np.finfo(float).tiny,
    )
    action_error_ratio = float(np.linalg.norm(action_error) / residual_norm)
    binding_passed = bool(
        block_relative["augmented"]
        <= CONTRACT["maximum_direct_augmented_JVP_relative_defect"]
        and max(block_relative.values())
        <= CONTRACT["maximum_block_JVP_relative_defect"]
        and max(central_five.values())
        <= CONTRACT["maximum_central_five_point_relative_defect"]
        and action_error_ratio
        <= CONTRACT["maximum_actual_correction_action_error_to_residual"]
        and linear_residual
        <= CONTRACT["maximum_equilibrated_linear_relative_residual"]
    )
    sweep_executed = False
    line_search_executed = False
    sweep_metrics = {}
    line_metrics = []
    if binding_passed:
        sweep_executed = True
        for step in DIAGNOSTIC_DIRECTIONAL_STEPS:
            current_central, current_five = _finite_difference(
                step,
                unknown,
                direction,
                data,
            )
            sweep_metrics[f"{step:.12e}"] = {
                "augmented_analytic_five_relative_defect": _relative(
                    analytic["augmented"], current_five["augmented"]
                ),
                "augmented_central_five_relative_defect": _relative(
                    current_central["augmented"], current_five["augmented"]
                ),
                "actual_action_error_to_residual": float(
                    np.linalg.norm(
                        correction_norm
                        * (current_five["augmented"] - analytic["augmented"])
                    )
                    / residual_norm
                ),
            }
        line_search_executed = True
        for alpha in LINE_SEARCH_ALPHAS:
            print(f"f24e3: line sample alpha={alpha:.8f}", flush=True)
            evaluation = _evaluate_unknown(unknown + alpha * correction, data)
            blocks = _blocks(evaluation, data["rows"])
            line_metrics.append(
                {
                    "alpha": alpha,
                    "augmented_L2_norm": float(
                        np.linalg.norm(blocks["augmented"])
                    ),
                    "augmented_maximum": float(
                        np.max(np.abs(blocks["augmented"]))
                    ),
                    "linearized_L2_norm": float(
                        np.linalg.norm(
                            base_blocks["augmented"]
                            + alpha * matrix.scaled_matrix @ correction
                        )
                    ),
                    "block_L2_norms": {
                        name: float(np.linalg.norm(values))
                        for name, values in blocks.items()
                    },
                }
            )
    discrepancy = analytic["augmented"] - five["augmented"]
    metrics = {
        "binding_directional_step": BINDING_DIRECTIONAL_STEP,
        "base_maximum_scaled_residual": float(
            np.max(np.abs(base_blocks["augmented"]))
        ),
        "base_residual_L2_norm": residual_norm,
        "correction_L2_norm": correction_norm,
        "equilibrated_linear_relative_residual": linear_residual,
        "direct_linear_relative_residual": direct_linear_residual,
        "equilibrated_direct_correction_relative_defect": correction_parity,
        "monolithic_matrix_reassembly_relative_defect": monolithic_parity,
        "block_analytic_five_relative_defects": block_relative,
        "block_central_five_relative_defects": central_five,
        "actual_correction_action_error_to_residual": action_error_ratio,
        "binding_action_passed": binding_passed,
        "sweep_executed": sweep_executed,
        "sweep": sweep_metrics,
        "line_search_executed": line_search_executed,
        "line_search": line_metrics,
        "discrepancy_localization": _component_localization(
            discrepancy,
            int(data["state"].shape[0]),
        ),
        "wall_seconds": time.perf_counter() - began,
    }
    if not binding_passed:
        classification = (
            "fixed_Q_endpoint_Newton_action_unresolved_"
            "derivative_precision_repair_required"
        )
        derivative_repair = True
        residual_floor_audit = False
    else:
        classification = (
            "fixed_Q_endpoint_matrix_action_certified_"
            "residual_merit_floor_audit_authorized"
        )
        derivative_repair = False
        residual_floor_audit = True
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": binding_passed,
        "analysis_only": True,
        "trajectory_executed": False,
        "physical_failure_detected": False,
        "parent_rejections_preserved": True,
        "endpoint_derivative_precision_repair_authorized": derivative_repair,
        "residual_merit_floor_audit_authorized": residual_floor_audit,
        "adaptive_refresh_implementation_authorized": False,
        "physical_history_ladder_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(RESULT_DIRECTORY / "summary.json", summary)
    _write(RESULT_DIRECTORY / "metrics.json", metrics)
    _write(RESULT_DIRECTORY / "contract.json", CONTRACT)
    _write_npz(
        RESULT_DIRECTORY / "decisive_arrays.npz",
        unknown=unknown,
        correction=correction,
        direction=direction,
        base_residual=base_blocks["augmented"],
        exact_matrix_action=analytic["augmented"],
        central_JVP=central["augmented"],
        five_point_JVP=five["augmented"],
        action_discrepancy=discrepancy,
        mapped_analytic_JVP=analytic["mapped_storage"],
        mapped_five_point_JVP=five["mapped_storage"],
        height_analytic_JVP=analytic["height_storage"],
        height_five_point_JVP=five["height_storage"],
        stationary_analytic_JVP=analytic["stationary"],
        stationary_five_point_JVP=five["stationary"],
        reaction_analytic_JVP=analytic["reaction"],
        reaction_five_point_JVP=five["reaction"],
        constraint_analytic_JVP=analytic["constraint"],
        constraint_five_point_JVP=five["constraint"],
    )
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
        "DIAGNOSTIC" if binding_passed else "REJECTED",
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
