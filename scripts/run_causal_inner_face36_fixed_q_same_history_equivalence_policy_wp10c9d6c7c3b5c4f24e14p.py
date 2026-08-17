#!/usr/bin/env python3
"""Certify the nonpropagating same-history endpoint-polish policy."""

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
e14h = importlib.import_module(  # noqa: E402
    "run_causal_inner_face36_fixed_q_exact_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14h"
)
e14n = importlib.import_module(  # noqa: E402
    "run_causal_inner_face36_fixed_q_cold_shadow_endpoint_diagnostic_"
    "wp10c9d6c7c3b5c4f24e14n"
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    _equilibrated_dense_solve,
    causal_five_field_fixed_q_augmented_step_matrix,
    load_causal_five_field_fixed_q_continuation_state,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14p"
ARTIFACT = (
    "causal_inner_face36_fixed_q_same_history_equivalence_policy_"
    "wp10c9d6c7c3b5c4f24e14p"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
MANIFEST_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_same_history_equivalence_policy_manifest_"
    "wp10c9d6c7c3b5c4f24e14o"
)
RETRY_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
TIMESTEP_SECONDS = 1.0e-7
PRODUCTION_RESIDUAL_TOLERANCE = 1.0e-10
EQUIVALENCE_RESIDUAL_TOLERANCE = 1.0e-12
STATE_EQUIVALENCE_TOLERANCE = 1.0e-8
ACTION_EQUIVALENCE_TOLERANCE = 1.0e-8
MAXIMUM_SCALED_CHANGE = 5.0e-3


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(e14h._plain(payload), indent=2, sort_keys=True) + "\n",
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
    _checksums(RETRY_DIRECTORY)
    summary = _read(MANIFEST_DIRECTORY / "summary.json")
    contract = _read(MANIFEST_DIRECTORY / "execution_contract.json")
    provenance = _read(MANIFEST_DIRECTORY / "provenance.json")
    policy = contract["equivalence_control_policy"]
    if (
        not summary["passed"]
        or not summary["same_history_equivalence_policy_certificate_authorized"]
        or summary["primary_evidence_aggregation_manifest_authorized"]
        or contract["production_step_acceptance"]["maximum_scaled_residual"]
        != PRODUCTION_RESIDUAL_TOLERANCE
        or policy["maximum_scaled_residual_before_state_action_comparison"]
        != EQUIVALENCE_RESIDUAL_TOLERANCE
        or policy["maximum_endpoint_polish_exact_assemblies"] != 1
        or policy["maximum_endpoint_polish_corrections"] != 1
        or policy["polished_control_may_define_history"]
    ):
        raise RuntimeError("e14o equivalence-policy authorization changed")
    for relative, digest in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != digest:
            raise RuntimeError(f"frozen policy source changed: {relative}")
    return {"summary": summary, "contract": contract, "provenance": provenance}


def _load_result(name: str) -> dict[str, np.ndarray]:
    with np.load(RETRY_DIRECTORY / name, allow_pickle=False) as source:
        return {key: np.array(source[key], copy=True) for key in source.files}


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
                    "scientific_status": "PASS" if summary["passed"] else "FAIL",
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
        raise RuntimeError("same-history policy certificate requires a clean tree")
    data = e14d.e1._state_data("primary_20ms")
    checkpoint = load_causal_five_field_fixed_q_continuation_state(
        RETRY_DIRECTORY / "checkpoint_warm_1.npz", data["context"]
    )
    warm = _load_result("result_warm_2.npz")
    cold = _load_result("result_cold_shadow.npz")
    cold_scaled_increment = (cold["primitive_increment"] / data["columns"]).ravel()
    cold_evaluation = e14h._evaluate(
        data, checkpoint, cold_scaled_increment, cold["multipliers"]
    )
    cold_residual = np.asarray(
        cold_evaluation.augmented_scaled_residual, dtype=float
    )
    cold_action = e14n._reaction_action(
        cold_evaluation, checkpoint, cold["multipliers"]
    )
    warm_action = np.asarray(warm["scaled_reaction_rate_action_per_s"], dtype=float)
    reproduction = {
        "cold_residual_bitwise": bool(
            np.array_equal(cold_residual, cold["augmented_scaled_residual"])
        ),
        "cold_reaction_action_bitwise": bool(
            np.array_equal(cold_action, cold["scaled_reaction_rate_action_per_s"])
        ),
    }
    if not all(reproduction.values()):
        raise RuntimeError("accepted cold control changed before policy certificate")
    initial_residual = float(np.max(np.abs(cold_residual)))
    initially_accepted = initial_residual <= PRODUCTION_RESIDUAL_TOLERANCE
    polish_required = initial_residual > EQUIVALENCE_RESIDUAL_TOLERANCE
    if not initially_accepted or not polish_required:
        raise RuntimeError("saved cold control no longer selects endpoint polish")
    exact_began = time.perf_counter()
    exact = causal_five_field_fixed_q_augmented_step_matrix(
        data["context"],
        checkpoint.current_primitive_charts,
        cold["primitive_charts"],
        cold["multipliers"],
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
        reaction=cold_evaluation.reaction,
    )
    exact_wall = time.perf_counter() - exact_began
    exact_matrix = np.asarray(exact.scaled_matrix, dtype=float)
    correction, linear_residual = _equilibrated_dense_solve(
        exact_matrix, -cold_residual
    )
    dimensions = cold_scaled_increment.size
    state_correction = correction[:dimensions]
    bound_alpha = 1.0
    positive = state_correction > 0.0
    negative = state_correction < 0.0
    if np.any(positive):
        bound_alpha = min(
            bound_alpha,
            float(
                np.min(
                    (MAXIMUM_SCALED_CHANGE - cold_scaled_increment[positive])
                    / state_correction[positive]
                )
            ),
        )
    if np.any(negative):
        bound_alpha = min(
            bound_alpha,
            float(
                np.min(
                    (-MAXIMUM_SCALED_CHANGE - cold_scaled_increment[negative])
                    / state_correction[negative]
                )
            ),
        )
    bound_alpha = (
        1.0
        if bound_alpha >= 1.0
        else min(1.0, max(0.0, 0.99 * bound_alpha))
    )
    initial_merit = float(np.linalg.norm(cold_residual))
    trials = []
    accepted = None
    accepted_evaluation = None
    for factor in manifest["contract"]["equivalence_control_policy"][
        "relative_line_search_factors"
    ]:
        alpha = bound_alpha * float(factor)
        trial_unknown = np.concatenate(
            (cold_scaled_increment, cold["multipliers"])
        ) + alpha * correction
        trial_evaluation = e14h._evaluate(
            data,
            checkpoint,
            trial_unknown[:dimensions],
            trial_unknown[dimensions:],
        )
        trial_residual = np.asarray(
            trial_evaluation.augmented_scaled_residual, dtype=float
        )
        trial = {
            "factor": factor,
            "alpha": alpha,
            "maximum_scaled_residual": float(np.max(np.abs(trial_residual))),
            "euclidean_merit": float(np.linalg.norm(trial_residual)),
        }
        trials.append(trial)
        if (
            trial["euclidean_merit"] < initial_merit
            or trial["maximum_scaled_residual"] <= EQUIVALENCE_RESIDUAL_TOLERANCE
        ):
            accepted = trial_unknown
            accepted_evaluation = trial_evaluation
            break
    audit = None
    state_defect = None
    action_defect = None
    corrected_action = None
    if accepted is not None:
        audit = e14h._candidate_audit(
            data,
            checkpoint,
            accepted[:dimensions],
            accepted[dimensions:],
            accepted_evaluation,
        )
        corrected_primitive = checkpoint.current_primitive_charts + (
            data["columns"].ravel() * accepted[:dimensions]
        ).reshape(checkpoint.current_primitive_charts.shape)
        corrected_action = e14n._reaction_action(
            accepted_evaluation, checkpoint, accepted[dimensions:]
        )
        state_defect = e14d._scaled_state_absolute(
            corrected_primitive, warm["primitive_charts"], data["columns"]
        )
        action_defect = e14d._relative(corrected_action, warm_action)
    policy_passed = bool(
        audit is not None
        and audit["passed"]
        and audit["maximum_scaled_residual"] <= EQUIVALENCE_RESIDUAL_TOLERANCE
        and state_defect <= STATE_EQUIVALENCE_TOLERANCE
        and action_defect <= ACTION_EQUIVALENCE_TOLERANCE
    )
    classification = (
        "same_history_equivalence_policy_certified"
        if policy_passed
        else "same_history_equivalence_policy_failed"
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": policy_passed,
        "certificate_executed": True,
        "trajectory_executed": False,
        "continuation_state_constructed": False,
        "historical_parent_classification_preserved": "bounded_continuation_failed",
        "production_step_acceptance_changed": False,
        "primary_evidence_aggregation_manifest_authorized": policy_passed,
        "primary_retry_authorized": False,
        "heldout_continuation_authorized": False,
        "operational_timestep_study_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    metrics = {
        "schema_version": 1,
        "saved_control_reproduction": reproduction,
        "accepted_control_before_polish": initially_accepted,
        "initial_control_maximum_scaled_residual": initial_residual,
        "polish_required": polish_required,
        "exact_jacobian_wall_seconds": exact_wall,
        "exact_linear_solve_relative_residual": linear_residual,
        "line_search_trials": trials,
        "polished_candidate_audit": audit,
        "polished_to_warm_comparison": {
            "scaled_state_absolute_defect": state_defect,
            "reaction_action_relative_defect": action_defect,
        },
        "policy_passed": policy_passed,
        "nonpropagation": {
            "continuation_state_constructed": False,
            "candidate_entered_history": False,
            "accepted_trajectory_horizon_seconds_added": 0.0,
        },
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(CANONICAL_DIRECTORY / "metrics.json", metrics)
    arrays = {
        "accepted_control_residual": cold_residual,
        "accepted_control_reaction_action": cold_action,
        "warm_reference_reaction_action": warm_action,
        "exact_matrix": exact_matrix,
        "exact_correction": correction,
    }
    if accepted is not None:
        arrays.update(
            {
                "polished_scaled_increment": accepted[:dimensions],
                "polished_multipliers": accepted[dimensions:],
                "polished_residual": np.asarray(
                    accepted_evaluation.augmented_scaled_residual, dtype=float
                ),
                "polished_reaction_action": corrected_action,
            }
        )
    with (CANONICAL_DIRECTORY / "certificate_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "manifest_summary_sha256": _sha(MANIFEST_DIRECTORY / "summary.json"),
            "source_hashes": manifest["provenance"]["source_hashes"],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name) for name in e14d.THREAD_ENVIRONMENT
            },
        },
    )
    files = (
        "certificate_arrays.npz",
        "metrics.json",
        "provenance.json",
        "summary.json",
    )
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
