#!/usr/bin/env python3
"""Run one nonpropagating exact-Jacobian correction at rejected warm_1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
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

e14d = importlib.import_module(  # noqa: E402
    "run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d"
)
from imri_qpe.layer3_minidisk_1d.causal_inner_bdf import (  # noqa: E402
    causal_bdf_coefficients,
    causal_bdf_weighted_increment,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    _equilibrated_dense_solve,
    _fixed_q_storage_parity_defect,
    _multiplier_weighted_action_ledger_defect,
    causal_five_field_fixed_q_augmented_step_matrix,
    evaluate_causal_five_field_fixed_q_bdf,
    load_causal_five_field_fixed_q_continuation_state,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14h"
ARTIFACT = (
    "causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14h"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
MANIFEST_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_exact_endpoint_diagnostic_manifest_"
    "wp10c9d6c7c3b5c4f24e14g"
)
E14D_DIRECTORY = e14d.CANONICAL_DIRECTORY
E14F_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_warm_failure_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14f"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
TIMESTEP_SECONDS = 1.0e-7
RESIDUAL_TOLERANCE = 1.0e-10
MAXIMUM_SCALED_CHANGE = 5.0e-3


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
        return float(value)
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
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_tree_is_clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(
            ("git", "diff", "--cached", "--quiet"), cwd=ROOT
        ).returncode
        == 0
    )


def _checksums(directory: Path) -> dict[str, str]:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")
    return entries


def _validate_manifest() -> dict:
    _checksums(MANIFEST_DIRECTORY)
    summary = _read(MANIFEST_DIRECTORY / "summary.json")
    contract = _read(MANIFEST_DIRECTORY / "execution_contract.json")
    provenance = _read(MANIFEST_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["exact_endpoint_diagnostic_execution_authorized"]
        or summary["warm_policy_execution_authorized"]
        or not contract["hard_stops"]["no_trajectory_advance"]
        or contract["authorized_diagnostic"][
            "maximum_exact_complete_jacobian_assemblies"
        ]
        != 1
        or contract["authorized_diagnostic"]["maximum_exact_newton_corrections"]
        != 1
    ):
        raise RuntimeError("e14g endpoint diagnostic authorization changed")
    for relative, digest in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != digest:
            raise RuntimeError(f"frozen endpoint source changed: {relative}")
    return {"summary": summary, "contract": contract, "provenance": provenance}


def _evaluate(
    data: dict,
    checkpoint,
    scaled_increment: np.ndarray,
    multipliers: np.ndarray,
    *,
    direct_rate: bool = False,
):
    old = checkpoint.current_primitive_charts
    new = old + (
        data["columns"].ravel() * scaled_increment
    ).reshape(old.shape)
    return evaluate_causal_five_field_fixed_q_bdf(
        old,
        new,
        multipliers,
        checkpoint.q3_target,
        TIMESTEP_SECONDS,
        data["context"],
        order=2,
        history=checkpoint.history,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        constraint_row_scales=checkpoint.constraint_row_scales,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=checkpoint.next_reaction_channel_transform,
        scaled_primitive_increment=scaled_increment,
        scaled_rate_per_s=(
            scaled_increment / TIMESTEP_SECONDS if direct_rate else None
        ),
        maximum_schur_condition_number=1.0e8,
    )


def _candidate_audit(
    data: dict,
    checkpoint,
    scaled_increment: np.ndarray,
    multipliers: np.ndarray,
    evaluation,
) -> dict:
    direct = _evaluate(
        data,
        checkpoint,
        scaled_increment,
        multipliers,
        direct_rate=True,
    )
    storage_parity = _fixed_q_storage_parity_defect(evaluation, direct)
    action_ledger = _multiplier_weighted_action_ledger_defect(
        evaluation.reaction,
        multipliers,
        checkpoint.next_reaction_channel_transform,
    )
    minimum_factor = min(
        float(
            evaluation.monolithic_evaluation.current_storage_increment
            .minimum_path_reconstruction_factor
        ),
        evaluation.reaction.minimum_q3_reconstruction_factor,
    )
    maximum_factor = max(
        float(
            evaluation.monolithic_evaluation.current_storage_increment
            .maximum_path_reconstruction_factor
        ),
        evaluation.reaction.maximum_q3_reconstruction_factor,
    )
    new = checkpoint.current_primitive_charts + (
        data["columns"].ravel() * scaled_increment
    ).reshape(checkpoint.current_primitive_charts.shape)
    physical = e14d.e1._state_audit(data["context"], new)
    maximum_residual = float(
        np.max(np.abs(evaluation.augmented_scaled_residual))
    )
    checks = {
        "complete_residual": maximum_residual <= RESIDUAL_TOLERANCE,
        "Q3": evaluation.maximum_constraint_relative_defect <= 1.0e-12,
        "incoming_excision": (
            evaluation.monolithic_evaluation.incoming_excision_characteristics
            == 0
        ),
        "storage_parity": storage_parity <= 1.0e-9,
        "reconstruction": (
            minimum_factor >= 1.0 - 1.0e-12
            and maximum_factor <= 1.0 + 1.0e-12
        ),
        "reaction_ledger": (
            evaluation.reaction.maximum_reaction_ledger_relative_defect
            <= 1.0e-12
        ),
        "constraint_action_ledger": action_ledger <= 1.0e-12,
        "primitive_change": (
            float(np.max(np.abs(scaled_increment))) <= MAXIMUM_SCALED_CHANGE
        ),
        "reaction_conditioning": (
            evaluation.reaction.raw_schur_numerical_rank == 3
            and evaluation.reaction.raw_schur_condition_number <= 1.0e8
            and evaluation.reaction.maximum_raw_schur_solve_relative_defect
            <= 1.0e-12
        ),
        "physical_height": float(physical["maximum_h_over_r"]) <= 0.12,
        "physical_optical_depth": (
            float(physical["minimum_scattering_optical_depth"]) >= 1.0
        ),
    }
    coefficients = causal_bdf_coefficients(
        2,
        TIMESTEP_SECONDS,
        checkpoint.history.previous_timestep_seconds,
    )
    previous_scaled_increment = (
        checkpoint.history.previous_primitive_increment / data["columns"]
    ).ravel()
    scaled_bdf_rate = causal_bdf_weighted_increment(
        scaled_increment,
        previous_scaled_increment,
        coefficients,
    ) / TIMESTEP_SECONDS
    reaction_action = (
        evaluation.reaction.raw_reaction_lift
        @ checkpoint.next_reaction_channel_transform
        @ multipliers
    )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failure_reasons": [name for name, passed in checks.items() if not passed],
        "maximum_scaled_residual": maximum_residual,
        "maximum_Q3_relative_defect": evaluation.maximum_constraint_relative_defect,
        "maximum_storage_parity_relative_defect": storage_parity,
        "minimum_path_reconstruction_factor": minimum_factor,
        "maximum_path_reconstruction_factor": maximum_factor,
        "maximum_reaction_ledger_relative_defect": (
            evaluation.reaction.maximum_reaction_ledger_relative_defect
        ),
        "maximum_constraint_action_ledger_relative_defect": action_ledger,
        "raw_Schur_rank": evaluation.reaction.raw_schur_numerical_rank,
        "raw_Schur_condition_number": (
            evaluation.reaction.raw_schur_condition_number
        ),
        "maximum_raw_Schur_solve_relative_defect": (
            evaluation.reaction.maximum_raw_schur_solve_relative_defect
        ),
        "maximum_scaled_primitive_change": float(
            np.max(np.abs(scaled_increment))
        ),
        "maximum_H_over_R": float(physical["maximum_h_over_r"]),
        "minimum_scattering_optical_depth": float(
            physical["minimum_scattering_optical_depth"]
        ),
        "incoming_excision_characteristics": (
            evaluation.monolithic_evaluation.incoming_excision_characteristics
        ),
        "scaled_bdf_rate_norm": float(np.linalg.norm(scaled_bdf_rate)),
        "scaled_reaction_action_norm": float(np.linalg.norm(reaction_action)),
    }


def _correction_metrics(
    exact_matrix: np.ndarray,
    carried_matrix: np.ndarray,
    exact_correction: np.ndarray,
    carried_correction: np.ndarray,
) -> dict:
    tiny = np.finfo(float).tiny
    dot = float(exact_correction @ carried_correction)
    exact_norm = float(np.linalg.norm(exact_correction))
    carried_norm = float(np.linalg.norm(carried_correction))
    cosine = float(np.clip(dot / max(exact_norm * carried_norm, tiny), -1.0, 1.0))

    def action_defect(direction: np.ndarray) -> float:
        exact_action = exact_matrix @ direction
        carried_action = carried_matrix @ direction
        return float(
            np.linalg.norm(exact_action - carried_action)
            / max(
                float(np.linalg.norm(exact_action)),
                float(np.linalg.norm(carried_action)),
                tiny,
            )
        )

    return {
        "correction_cosine": cosine,
        "correction_angle_radians": float(np.arccos(cosine)),
        "exact_to_carried_correction_norm_ratio": exact_norm
        / max(carried_norm, tiny),
        "exact_jacobian_action_defect_on_carried_correction": action_defect(
            carried_correction
        ),
        "carried_matrix_action_defect_on_exact_correction": action_defect(
            exact_correction
        ),
        "full_matrix_relative_frobenius_defect_diagnostic_only": float(
            np.linalg.norm(exact_matrix - carried_matrix)
            / max(
                float(np.linalg.norm(exact_matrix)),
                float(np.linalg.norm(carried_matrix)),
                tiny,
            )
        ),
    }


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": (
                        "PASS" if summary["passed"] else "FAIL"
                    ),
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
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


def _run() -> dict:
    manifest = _validate_manifest()
    if not _tracked_tree_is_clean():
        raise RuntimeError("exact endpoint diagnostic requires a clean tree")
    data = e14d.e1._state_data("primary_20ms")
    checkpoint = load_causal_five_field_fixed_q_continuation_state(
        E14D_DIRECTORY / "checkpoint_cold_1.npz", data["context"]
    )
    with np.load(E14F_DIRECTORY / "endpoint_replay_arrays.npz", allow_pickle=False) as source:
        replay_arrays = {name: np.array(source[name], copy=True) for name in source.files}
    scaled_increment = (
        replay_arrays["rejected_primitive_increment"] / data["columns"]
    ).ravel()
    multipliers = replay_arrays["rejected_multipliers"]
    evaluation = _evaluate(data, checkpoint, scaled_increment, multipliers)
    residual = np.asarray(evaluation.augmented_scaled_residual, dtype=float)
    if not np.array_equal(
        residual, replay_arrays["committed_augmented_scaled_residual"]
    ):
        raise RuntimeError("endpoint residual replay changed before diagnosis")
    dimensions = scaled_increment.size
    exact_began = time.perf_counter()
    exact = causal_five_field_fixed_q_augmented_step_matrix(
        data["context"],
        checkpoint.current_primitive_charts,
        replay_arrays["rejected_primitive_charts"],
        multipliers,
        TIMESTEP_SECONDS,
        checkpoint.history.previous_timestep_seconds,
        order=2,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        constraint_row_scales=checkpoint.constraint_row_scales,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=checkpoint.next_reaction_channel_transform,
        reaction=evaluation.reaction,
    )
    exact_wall = time.perf_counter() - exact_began
    exact_matrix = np.asarray(exact.scaled_matrix, dtype=float)
    carried_matrix = np.array(
        replay_arrays["rejected_raw_solver_matrix"], copy=True
    )
    carried_matrix[:, dimensions:] = (
        carried_matrix[:, dimensions:]
        @ checkpoint.next_reaction_channel_transform
    )
    exact_correction, exact_linear_residual = _equilibrated_dense_solve(
        exact_matrix, -residual
    )
    carried_correction, carried_linear_residual = _equilibrated_dense_solve(
        carried_matrix, -residual
    )
    comparison = _correction_metrics(
        exact_matrix,
        carried_matrix,
        exact_correction,
        carried_correction,
    )
    state = scaled_increment
    state_correction = exact_correction[:dimensions]
    bound_alpha = 1.0
    positive = state_correction > 0.0
    negative = state_correction < 0.0
    if np.any(positive):
        bound_alpha = min(
            bound_alpha,
            float(
                np.min(
                    (MAXIMUM_SCALED_CHANGE - state[positive])
                    / state_correction[positive]
                )
            ),
        )
    if np.any(negative):
        bound_alpha = min(
            bound_alpha,
            float(
                np.min(
                    (-MAXIMUM_SCALED_CHANGE - state[negative])
                    / state_correction[negative]
                )
            ),
        )
    bound_alpha = (
        1.0
        if bound_alpha >= 1.0
        else min(1.0, max(0.0, 0.99 * bound_alpha))
    )
    merit = float(np.linalg.norm(residual))
    trials = []
    accepted = None
    accepted_evaluation = None
    factors = manifest["contract"]["authorized_diagnostic"][
        "relative_line_search_factors"
    ]
    for factor in factors:
        alpha = bound_alpha * float(factor)
        trial_unknown = np.concatenate((scaled_increment, multipliers)) + (
            alpha * exact_correction
        )
        trial_evaluation = _evaluate(
            data,
            checkpoint,
            trial_unknown[:dimensions],
            trial_unknown[dimensions:],
        )
        trial_residual = trial_evaluation.augmented_scaled_residual
        trial_metrics = {
            "factor": factor,
            "alpha": alpha,
            "maximum_scaled_residual": float(
                np.max(np.abs(trial_residual))
            ),
            "euclidean_merit": float(np.linalg.norm(trial_residual)),
        }
        trials.append(trial_metrics)
        if (
            trial_metrics["euclidean_merit"] < merit
            or trial_metrics["maximum_scaled_residual"] <= RESIDUAL_TOLERANCE
        ):
            accepted = trial_unknown
            accepted_evaluation = trial_evaluation
            break
    audit = None
    if accepted is not None:
        audit = _candidate_audit(
            data,
            checkpoint,
            accepted[:dimensions],
            accepted[dimensions:],
            accepted_evaluation,
        )
    improved = bool(trials and min(t["euclidean_merit"] for t in trials) < merit)
    positive_diagnosis = bool(audit is not None and audit["passed"])
    if positive_diagnosis:
        classification = "stale_carried_matrix_refresh_trigger_diagnosed"
    elif improved:
        classification = "endpoint_exact_diagnostic_inconclusive"
    else:
        classification = "endpoint_exact_diagnostic_failed"
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": positive_diagnosis,
        "diagnostic_executed": True,
        "trajectory_executed": False,
        "continuation_state_constructed": False,
        "exact_jacobian_assemblies": 1,
        "exact_newton_corrections": 1,
        "parent_classification_preserved": "bounded_continuation_failed",
        "warm_policy_manifest_authorized": positive_diagnosis,
        "warm_policy_execution_authorized": False,
        "full_primary_retry_authorized": False,
        "heldout_continuation_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    metrics = {
        "schema_version": 1,
        "initial_maximum_scaled_residual": float(np.max(np.abs(residual))),
        "initial_euclidean_merit": merit,
        "exact_jacobian_wall_seconds": exact_wall,
        "exact_linear_solve_relative_residual": exact_linear_residual,
        "carried_linear_solve_relative_residual": carried_linear_residual,
        "bound_limited_initial_alpha": bound_alpha,
        "matrix_and_correction_comparison": comparison,
        "line_search_trials": trials,
        "corrected_candidate_audit": audit,
        "positive_diagnosis": positive_diagnosis,
        "nonpropagation": {
            "continuation_state_constructed": False,
            "candidate_entered_history": False,
            "accepted_trajectory_horizon_seconds_added": 0.0,
        },
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(CANONICAL_DIRECTORY / "metrics.json", metrics)
    with (CANONICAL_DIRECTORY / "diagnostic_arrays.npz").open("wb") as handle:
        arrays = {
            "initial_residual": residual,
            "exact_matrix": exact_matrix,
            "carried_matrix_rebased": carried_matrix,
            "exact_correction": exact_correction,
            "carried_correction": carried_correction,
        }
        if accepted is not None:
            arrays.update(
                {
                    "corrected_scaled_increment": accepted[:dimensions],
                    "corrected_multipliers": accepted[dimensions:],
                    "corrected_residual": (
                        accepted_evaluation.augmented_scaled_residual
                    ),
                }
            )
        np.savez_compressed(handle, **arrays)
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "manifest_sha256": _sha(MANIFEST_DIRECTORY / "summary.json"),
            "source_hashes": manifest["provenance"]["source_hashes"],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name) for name in e14d.THREAD_ENVIRONMENT
            },
        },
    )
    files = ("diagnostic_arrays.npz", "metrics.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in files),
        encoding="utf-8",
    )
    _catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        raise SystemExit("select --run")
    print(json.dumps(_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
