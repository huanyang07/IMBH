#!/usr/bin/env python3
"""Freeze the primary fixed-Q bounded-continuation execution manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14c"
ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_bounded_continuation_manifest_"
    "wp10c9d6c7c3b5c4f24e14c"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
E14A_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_bounded_continuation_cost_manifest_"
    "wp10c9d6c7c3b5c4f24e14a"
)
E14B_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_continuation_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14b"
)
SEED_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
CANONICAL_SEED = E14B_DIRECTORY / "canonical_seed_continuation.npz"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "manifest_wp10c9d6c7c3b5c4f24e14c.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "manifest_wp10c9d6c7c3b5c4f24e14c.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
    "scripts/run_causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1.py",
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
TIMESTEP_SECONDS = 1.0e-7


ROOT_SEQUENCE = (
    {
        "index": 1,
        "label": "cold_1",
        "start_checkpoint": "canonical_seed",
        "solver_mode": "cold",
        "initial_exact_complete_matrix_required": True,
        "maximum_exact_assemblies": 2,
        "may_define_main_history": True,
    },
    {
        "index": 2,
        "label": "warm_1",
        "start_checkpoint": "cold_1",
        "solver_mode": "carried_raw_coordinate_Broyden",
        "initial_exact_complete_matrix_required": False,
        "maximum_exact_assemblies": 1,
        "may_define_main_history": True,
    },
    {
        "index": 3,
        "label": "warm_2",
        "start_checkpoint": "warm_1",
        "solver_mode": "carried_raw_coordinate_Broyden",
        "initial_exact_complete_matrix_required": False,
        "maximum_exact_assemblies": 1,
        "may_define_main_history": True,
    },
    {
        "index": 4,
        "label": "warm_3",
        "start_checkpoint": "warm_2",
        "solver_mode": "carried_raw_coordinate_Broyden",
        "initial_exact_complete_matrix_required": False,
        "maximum_exact_assemblies": 1,
        "may_define_main_history": True,
    },
)


CONTRACT = {
    "schema_version": 1,
    "work_package": WORK_PACKAGE,
    "definitions_only": True,
    "trajectory_may_execute_during_freeze": False,
    "parent_work_packages": [
        "WP10c9d6c7c3b5c4f24e14a",
        "WP10c9d6c7c3b5c4f24e14b",
    ],
    "authorized_execution": {
        "case": "primary_20ms",
        "layout": "middle",
        "fixed_Q_coordinates": ["M", "J", "E"],
        "timestep_seconds": TIMESTEP_SECONDS,
        "new_main_BDF2_roots": 4,
        "new_main_horizon_seconds": 4.0e-7,
        "interpretation": "continuation_infrastructure_and_cost_only",
    },
    "seed_contract": {
        "continuation_artifact": E14B_DIRECTORY.name,
        "continuation_file": "canonical_seed_continuation.npz",
        "seed_state_artifact": SEED_DIRECTORY.name,
        "current_order": 2,
        "next_order": 2,
        "completed_steps": 2,
        "previous_timestep_seconds": TIMESTEP_SECONDS,
        "complete_history_required": True,
        "carried_nonlinear_matrix_present": False,
        "first_new_root_must_be_cold": True,
        "synthetic_or_projected_history_forbidden": True,
    },
    "predictor_contract": {
        "state_rate_predictor": (
            "checkpoint_previous_primitive_increment_divided_by_"
            "primitive_column_scales_and_previous_timestep"
        ),
        "state_rate_predictor_must_be_reconstructed_identically_after_restart": (
            True
        ),
        "raw_multiplier_predictor": "checkpoint_raw_multiplier_predictor",
        "step_multiplier_predictor": (
            "solve_current_reaction_transform_times_lambda_equals_raw_mu"
        ),
        "multiplier_coordinate_equality_is_binding": False,
        "physical_reaction_action_invariance_is_binding": True,
    },
    "main_root_sequence": list(ROOT_SEQUENCE),
    "solver_contract": {
        "binding_temporal_form": "increment_primary",
        "direct_rate_form": "post_root_parity_audit_only",
        "reaction_channel_basis": "frozen_normalized",
        "serialized_solver_matrix_basis": "raw_reaction_channels",
        "warm_matrix_multiplier_columns_rebased_to_current_transform": True,
        "warm_matrix_anchor_target_scale_hashes_must_match": True,
        "warm_refresh_policy": "after_complete_line_search_failure_only",
        "maximum_newton_iterations": 8,
        "maximum_line_search_trials_per_iteration": 12,
        "accepted_history_only": True,
        "production_defaults_may_change": False,
    },
    "restart_replay_contract": {
        "warm_root_indices_are_one_based": True,
        "restart_checkpoint": "after_warm_1_before_warm_2",
        "replayed_suffix": ["warm_2", "warm_3"],
        "bitwise_fields": [
            "primitive_state",
            "primitive_increment",
            "mapped_storage_history",
            "responsive_height_history",
            "Q3_target_and_scales",
            "raw_multiplier_predictor",
            "raw_coordinate_Broyden_matrix_and_anchor",
            "reaction_action",
            "accepted_and_rejected_line_search_trace",
            "decisive_diagnostics",
        ],
        "profiling_times_are_excluded_from_bitwise_comparison": True,
    },
    "same_history_cold_shadow": {
        "main_root": "warm_2",
        "start_checkpoint": "after_warm_1",
        "may_define_main_history": False,
        "initial_exact_complete_matrix_required": True,
        "maximum_exact_assemblies": 2,
        "maximum_scaled_state_difference": 1.0e-8,
        "maximum_reaction_action_relative_difference": 1.0e-8,
        "maximum_warm_to_cold_wall_time_ratio": 0.75,
    },
    "matched_endpoint_half_step_audit": {
        "conditional_on_main_and_replay_scientific_pass": True,
        "full_step_root": "warm_3",
        "start_checkpoint": "after_warm_2",
        "half_timestep_seconds": 5.0e-8,
        "number_of_cold_half_steps": 2,
        "variable_step_BDF2_coefficients_required": True,
        "each_half_step_initial_exact_complete_matrix_required": True,
        "maximum_exact_assemblies_per_half_step": 2,
        "half_steps_may_define_main_history": False,
        "maximum_state_difference_relative_to_full_step_change": 0.1,
        "maximum_reaction_action_relative_difference": 0.1,
    },
    "inherited_step_gates": {
        "maximum_scaled_residual": 1.0e-10,
        "maximum_Q3_relative_defect": 1.0e-12,
        "maximum_ledger_relative_defect": 1.0e-12,
        "maximum_storage_parity_relative_defect": 1.0e-9,
        "minimum_path_reconstruction_factor": 1.0 - 1.0e-12,
        "maximum_path_reconstruction_factor": 1.0 + 1.0e-12,
        "raw_Schur_rank": 3,
        "maximum_raw_Schur_condition_number": 1.0e8,
        "maximum_H_over_R": 0.12,
        "minimum_scattering_optical_depth": 1.0,
        "maximum_scaled_primitive_change": 5.0e-3,
        "incoming_excision_characteristics": 0,
    },
    "trajectory_gates": {
        "maximum_cumulative_absolute_ledger_budget": 4.0e-12,
        "cumulative_ledger_definition": (
            "sum_over_four_main_roots_of_the_maximum_of_reaction_ledger_"
            "and_multiplier_weighted_constraint_action_ledger_defects"
        ),
        "checkpoint_every_accepted_main_endpoint": True,
        "checkpoint_roundtrip_bitwise_each_main_endpoint": True,
        "minimum_warm_roots_without_exact_refresh": 2,
        "all_main_roots_must_pass_before_any_nonpropagating_audit": True,
    },
    "profiling_contract": {
        "thread_environment": {
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "root_total_wall_and_process_time_required": True,
        "solver_activity_wall_counters_required": [
            "residual",
            "monolithic_residual",
            "reaction_construction",
            "descriptor_assembly",
            "descriptor_sparse_LU",
            "exact_complete_Jacobian",
            "bordered_linear_solve",
            "line_search_residual",
            "physical_acceptance",
        ],
        "checkpoint_read_write_wall_and_process_time_required": True,
        "event_trace_required": True,
        "record_function_evaluations": True,
        "record_exact_assemblies": True,
        "record_Broyden_updates": True,
        "record_linear_solves": True,
        "record_line_search_trials": True,
        "record_failed_line_search_trials": True,
        "record_matrix_age": True,
        "record_residual_margin": True,
        "record_checkpoint_bytes": True,
        "nested_activity_counters_may_not_be_summed_as_exclusive_time": True,
    },
    "classification_contract": {
        "scientific_and_cost_pass": "bounded_continuation_and_reuse_passed",
        "scientific_pass_cost_fail": (
            "bounded_continuation_valid_cost_failed"
        ),
        "scientific_fail": "bounded_continuation_failed",
        "cost_failure_may_not_override_scientific_pass": True,
        "scientific_failure_may_not_be_reclassified_as_cost_failure": True,
    },
    "execution_order": [
        "validate_committed_manifest_sources_parents_seed_and_thread_env",
        "load_and_bitwise_roundtrip_canonical_seed",
        "run_cold_1_then_warm_1_then_warm_2_then_warm_3_fail_closed",
        "restart_after_warm_1_and_replay_warm_2_then_warm_3_bitwise",
        "run_same_history_cold_shadow_for_warm_2_without_propagation",
        "run_two_cold_half_steps_from_warm_3_start_without_propagation",
        "classify_scientific_validity_before_cost",
        "canonicalize_all_pass_or_failure_evidence",
    ],
    "hard_stops": {
        "no_gate_relaxation": True,
        "no_undeclared_retry": True,
        "no_stale_or_optional_local_predictor": True,
        "no_rejected_step_in_history": True,
        "no_heldout_continuation": True,
        "no_operational_timestep_search": True,
        "no_fixed_Q_micro_solver": True,
        "no_physical_microburst": True,
        "no_fast_averaging": True,
        "no_reduced_slow_evolution": True,
    },
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            ("git", "diff", "--cached", "--quiet"),
            cwd=ROOT,
        ).returncode
        == 0
    )


def _validate_checksums(directory: Path) -> dict[str, str]:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")
    return entries


def _parent_authorization() -> dict:
    e14a_hashes = _validate_checksums(E14A_DIRECTORY)
    e14b_hashes = _validate_checksums(E14B_DIRECTORY)
    seed_hashes = _validate_checksums(SEED_DIRECTORY)
    e14a = _read(E14A_DIRECTORY / "summary.json")
    e14b = _read(E14B_DIRECTORY / "summary.json")
    if (
        not e14a["passed"]
        or not e14a["implementation_preflight_authorized"]
        or e14a["bounded_continuation_execution_authorized"]
        or not e14b["passed"]
        or not e14b["primary_pilot_execution_manifest_authorized"]
        or e14b["bounded_continuation_execution_authorized"]
        or e14b["heldout_continuation_authorized"]
        or e14b["fixed_Q_micro_solver_authorized"]
        or e14b["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("primary bounded-continuation authorization changed")
    return {
        "e14a_summary": e14a,
        "e14b_summary": e14b,
        "e14a_hashes": e14a_hashes,
        "e14b_hashes": e14b_hashes,
        "seed_hashes": seed_hashes,
    }


def _seed_lock() -> dict:
    with np.load(CANONICAL_SEED, allow_pickle=False) as source:
        required = {
            "current_primitive_charts",
            "previous_primitive_charts",
            "previous_primitive_increment",
            "previous_mapped_storage_increment",
            "previous_responsive_height_storage_increment",
            "previous_timestep_seconds",
            "q3_target",
            "constraint_row_scales",
            "raw_multiplier_predictor",
            "next_reaction_channel_basis",
            "next_reaction_channel_transform",
            "completed_steps",
            "current_order",
            "next_order",
            "has_nonlinear_solver_state",
            "provenance_json",
        }
        if not required.issubset(source.files):
            raise RuntimeError("canonical continuation seed inventory changed")
        current = np.asarray(source["current_primitive_charts"], dtype=float)
        previous = np.asarray(source["previous_primitive_charts"], dtype=float)
        increment = np.asarray(source["previous_primitive_increment"], dtype=float)
        if not np.array_equal(current, previous + increment):
            raise RuntimeError("canonical continuation primitive history changed")
        payload = {
            "schema_version": 1,
            "path": str(CANONICAL_SEED.relative_to(ROOT)),
            "sha256": _sha(CANONICAL_SEED),
            "bytes": CANONICAL_SEED.stat().st_size,
            "array_inventory": sorted(source.files),
            "primitive_shape": list(current.shape),
            "previous_timestep_seconds": float(
                source["previous_timestep_seconds"]
            ),
            "completed_steps": int(source["completed_steps"]),
            "current_order": int(source["current_order"]),
            "next_order": int(source["next_order"]),
            "has_nonlinear_solver_state": bool(
                source["has_nonlinear_solver_state"]
            ),
            "next_reaction_channel_basis": str(
                source["next_reaction_channel_basis"].item()
            ),
            "primitive_history_bitwise": True,
            "provenance": json.loads(str(source["provenance_json"].item())),
        }
    if (
        payload["primitive_shape"] != [112, 5]
        or payload["previous_timestep_seconds"] != TIMESTEP_SECONDS
        or payload["completed_steps"] != 2
        or payload["current_order"] != 2
        or payload["next_order"] != 2
        or payload["has_nonlinear_solver_state"]
        or payload["next_reaction_channel_basis"] != "frozen_normalized"
    ):
        raise RuntimeError("canonical continuation seed semantics changed")
    return payload


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
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
    catalog["artifacts"][ARTIFACT] = {
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
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


def _freeze() -> dict:
    parent = _parent_authorization()
    seed = _seed_lock()
    if not _tracked_tree_is_clean():
        raise RuntimeError("primary continuation manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "primary_bounded_fixed_Q_continuation_manifest_frozen_"
            "execution_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "primary_bounded_continuation_execution_authorized": True,
        "heldout_continuation_authorized": False,
        "operational_timestep_study_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next_action": (
            "execute_only_the_frozen_primary_bounded_continuation_pilot"
        ),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(ARTIFACT_DIRECTORY / "execution_manifest.json", CONTRACT)
    _write(ARTIFACT_DIRECTORY / "parent_authorization.json", parent)
    _write(ARTIFACT_DIRECTORY / "seed_lock.json", seed)
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in SOURCE_FILES
            },
            "e14a_summary_sha256": _sha(E14A_DIRECTORY / "summary.json"),
            "e14b_summary_sha256": _sha(E14B_DIRECTORY / "summary.json"),
            "canonical_seed_sha256": seed["sha256"],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment_at_freeze": {
                name: os.environ.get(name)
                for name in CONTRACT["profiling_contract"][
                    "thread_environment"
                ]
            },
        },
    )
    files = (
        "execution_manifest.json",
        "parent_authorization.json",
        "provenance.json",
        "seed_lock.json",
        "summary.json",
    )
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in files
        ),
        encoding="utf-8",
    )
    _catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        raise SystemExit("select --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
