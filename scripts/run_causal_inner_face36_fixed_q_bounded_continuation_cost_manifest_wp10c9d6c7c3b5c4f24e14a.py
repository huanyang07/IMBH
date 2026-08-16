#!/usr/bin/env python3
"""Freeze the definitions-only bounded fixed-Q continuation/cost manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14a"
ARTIFACT = (
    "causal_inner_face36_fixed_q_bounded_continuation_cost_manifest_"
    "wp10c9d6c7c3b5c4f24e14a"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
PARENT_ARTIFACT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_hardened_"
    "wp10c9d6c7c3b5c4f24e13a"
)
SEED_ARTIFACT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_bounded_continuation_cost_"
    "manifest_wp10c9d6c7c3b5c4f24e14a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_bounded_continuation_cost_"
    "manifest_wp10c9d6c7c3b5c4f24e14a.py"
)
FIXED_Q_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py"
)
MONOLITHIC_BDF_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

ROOT_ARRAYS = (
    "bdf1_primitive_charts",
    "bdf1_primitive_increment",
    "bdf1_scaled_rate_per_s",
    "bdf1_multipliers",
    "bdf2_primitive_charts",
    "bdf2_primitive_increment",
    "bdf2_scaled_rate_per_s",
    "bdf2_multipliers",
    "bdf2_scaled_reaction_rate_action_per_s",
)
COMPLETE_HISTORY_ARRAYS = (
    "bdf2_previous_mapped_storage_increment",
    "bdf2_previous_responsive_height_storage_increment",
    "bdf2_previous_timestep_seconds",
)

CONTRACT = {
    "schema_version": 1,
    "parent_work_package": "WP10c9d6c7c3b5c4f24e13a",
    "definitions_only": True,
    "trajectory_may_execute": False,
    "authorized_objective": (
        "implementation_only_fixed_Q_continuation_state_and_solver_reuse_"
        "preflight"
    ),
    "parent_metadata_roles": {
        "contract.json": "input_pre_ladder_execution_contract",
        "summary.json": "post_ladder_final_authorization",
    },
    "seed": {
        "layout": "middle",
        "state": "primary_20ms",
        "timestep_seconds": 1.0e-7,
        "canonical_artifact": SEED_ARTIFACT.name,
        "required_root_arrays": list(ROOT_ARRAYS),
        "required_complete_history_arrays": list(COMPLETE_HISTORY_ARRAYS),
        "complete_history_policy": (
            "recompute_the_declared_straight_primitive_path_storage_integrals_"
            "from_hash_validated_canonical_BDF1_BDF2_states_and_freeze_them;_"
            "if_exact_reconstruction_cannot_be_certified_rerun_one_authentic_"
            "coarse_startup"
        ),
        "synthetic_or_projected_history_forbidden": True,
    },
    "implementation_contract": {
        "continuation_state_valid_after_orders": [1, 2],
        "rejected_step_may_define_history": False,
        "persist_current_and_previous_primitive_states": True,
        "persist_complete_mapped_storage_history": True,
        "persist_complete_responsive_height_history": True,
        "persist_current_and_previous_timesteps": True,
        "persist_Q3_target_and_constraint_scales": True,
        "persist_multiplier_predictor": True,
        "persist_reaction_basis_and_transform": True,
        "persist_final_bordered_Broyden_matrix": True,
        "persist_matrix_anchor_order_timestep_age_and_hashes": True,
        "bitwise_save_load_required": True,
        "production_defaults_may_change": False,
    },
    "reaction_coordinate_contract": {
        "serialized_solver_matrix_multiplier_basis": "raw_reaction_channels",
        "step_residual_multiplier_basis": "frozen_normalized",
        "physical_raw_coefficients": "mu_equals_T_lambda",
        "predictor_rebase": (
            "lambda_new_equals_inverse_T_new_times_T_old_times_lambda_old"
        ),
        "matrix_multiplier_column_rebase": (
            "J_lambda_new_equals_J_lambda_old_times_inverse_T_old_times_T_new"
        ),
        "physical_reaction_action_invariance_required": True,
        "multiplier_coordinate_equality_binding": False,
    },
    "prospective_primary_pilot": {
        "execution_authorized_by_this_manifest": False,
        "new_BDF2_roots": 4,
        "cold_roots": 1,
        "warm_roots": 3,
        "new_physical_horizon_seconds": 4.0e-7,
        "first_root_requires_initial_exact_matrix": True,
        "warm_roots_force_initial_exact_matrix": False,
        "warm_refresh_only_after_complete_line_search_failure": True,
        "maximum_warm_exact_refreshes_per_root": 1,
        "checkpoint_every_accepted_endpoint": True,
        "restart_before_warm_root_index": 2,
        "bitwise_two_step_suffix_replay_required": True,
        "same_history_cold_shadow_at_warm_root_index": 2,
        "cold_shadow_may_define_history": False,
        "conditional_matched_endpoint_half_step_audit": True,
        "pilot_interpretation": "infrastructure_and_cost_only",
    },
    "profiling_contract": {
        "pin_thread_environment": True,
        "record_wall_and_process_time": True,
        "exclusive_timing_categories": [
            "monolithic_residual",
            "fixed_Q_reaction_construction",
            "descriptor_assembly_and_sparse_LU",
            "exact_complete_Jacobian",
            "bordered_linear_solve",
            "line_search_residuals",
            "physical_acceptance_audit",
            "checkpoint_write",
            "checkpoint_read",
        ],
        "record_function_evaluations": True,
        "record_exact_assemblies": True,
        "record_Broyden_updates": True,
        "record_failed_line_search_trials": True,
        "record_matrix_age": True,
        "record_residual_margin": True,
        "record_checkpoint_bytes": True,
        "same_history_cost_control_required": True,
    },
    "inherited_step_gates": {
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
        "incoming_excision_characteristics": 0,
    },
    "prospective_decision_gates": {
        "maximum_warm_cold_scaled_state_difference": 1.0e-8,
        "maximum_warm_cold_reaction_action_relative_difference": 1.0e-8,
        "minimum_warm_roots_without_exact_refresh": 2,
        "maximum_same_history_warm_to_cold_wall_time_ratio": 0.75,
        "maximum_half_step_state_difference_relative_to_full_step_change": 0.1,
        "maximum_half_step_reaction_action_relative_difference": 0.1,
        "maximum_cumulative_absolute_ledger_budget": 4.0e-12,
    },
    "result_classifications": {
        "scientific_and_cost_pass": "bounded_continuation_and_reuse_passed",
        "scientific_pass_cost_fail": (
            "bounded_continuation_valid_cost_failed"
        ),
        "scientific_fail": "bounded_continuation_failed",
    },
    "authorization_boundaries": {
        "bounded_continuation_execution_authorized": False,
        "heldout_continuation_authorized": False,
        "operational_timestep_study_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
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
            ("git", "diff", "--cached", "--quiet"), cwd=ROOT
        ).returncode
        == 0
    )


def _validate_checksums(directory: Path, required: set[str]) -> None:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    if set(entries) != required:
        raise RuntimeError(f"canonical checksum inventory changed: {directory}")
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")


def _npz_array_names(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path) as archive:
        return tuple(
            sorted(
                name.removesuffix(".npy")
                for name in archive.namelist()
                if name.endswith(".npy")
            )
        )


def _validate_parent_and_seed() -> tuple[dict, dict, dict]:
    _validate_checksums(
        PARENT_ARTIFACT,
        {"contract.json", "provenance.json", "summary.json"},
    )
    parent_summary = _read(PARENT_ARTIFACT / "summary.json")
    parent_contract = _read(PARENT_ARTIFACT / "contract.json")
    if (
        not parent_summary["passed"]
        or not parent_summary["one_Q_execution_manifest_authorized"]
        or parent_summary["fixed_Q_micro_solver_authorized"]
        or parent_summary["reduced_slow_evolution_authorized"]
        or parent_contract["one_Q_execution_manifest_authorized"]
    ):
        raise RuntimeError("fixed-Q refined-ladder authorization changed")
    _validate_checksums(
        SEED_ARTIFACT,
        {
            "contract.json",
            "decisive_arrays.npz",
            "metrics.json",
            "provenance.json",
            "summary.json",
        },
    )
    seed_summary = _read(SEED_ARTIFACT / "summary.json")
    seed_metrics = _read(SEED_ARTIFACT / "metrics.json")
    if (
        not seed_summary["passed"]
        or not seed_summary["primary_nonregression_passed"]
        or not seed_metrics["passed"]
        or not seed_metrics["BDF2_replay_bitwise"]
        or not seed_metrics["BDF1"]["accepted"]
        or not seed_metrics["BDF2"]["accepted"]
    ):
        raise RuntimeError("fixed-Q primary coarse seed changed")
    names = set(_npz_array_names(SEED_ARTIFACT / "decisive_arrays.npz"))
    if not set(ROOT_ARRAYS).issubset(names):
        raise RuntimeError("fixed-Q canonical root arrays are incomplete")
    inventory = {
        "schema_version": 1,
        "canonical_artifact": SEED_ARTIFACT.name,
        "available_arrays": sorted(names),
        "required_root_arrays": list(ROOT_ARRAYS),
        "required_complete_history_arrays": list(COMPLETE_HISTORY_ARRAYS),
        "canonical_root_arrays_complete": True,
        "complete_storage_history_present": set(COMPLETE_HISTORY_ARRAYS).issubset(
            names
        ),
        "resolution_required_before_execution": (
            "exact_path_reconstruction_certificate_or_authentic_startup_rerun"
        ),
    }
    return parent_summary, parent_contract, inventory


def _source_hashes() -> dict[str, str]:
    paths = (THIS_RUNNER, THIS_TEST, FIXED_Q_SOURCE, MONOLITHIC_BDF_SOURCE)
    return {path: _sha(ROOT / path) for path in paths}


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: _sha(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _catalog(directory: Path, summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(directory.iterdir()):
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


def _freeze() -> dict:
    parent_summary, parent_contract, inventory = _validate_parent_and_seed()
    if not _tracked_tree_is_clean():
        raise RuntimeError("bounded continuation manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_bounded_continuation_cost_manifest_frozen_"
            "implementation_preflight_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "parent_certificate_passed": True,
        "implementation_preflight_authorized": True,
        "bounded_continuation_execution_authorized": False,
        "heldout_continuation_authorized": False,
        "operational_timestep_study_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next_artifact": CONTRACT["authorized_objective"],
        "seed_complete_history_resolution_required": True,
    }
    authorization = {
        "schema_version": 1,
        "parent_artifact": PARENT_ARTIFACT.name,
        "input_execution_contract": {
            "path": str((PARENT_ARTIFACT / "contract.json").relative_to(ROOT)),
            "sha256": _sha(PARENT_ARTIFACT / "contract.json"),
            "role": CONTRACT["parent_metadata_roles"]["contract.json"],
            "one_Q_execution_manifest_authorized": parent_contract[
                "one_Q_execution_manifest_authorized"
            ],
        },
        "final_decision": {
            "path": str((PARENT_ARTIFACT / "summary.json").relative_to(ROOT)),
            "sha256": _sha(PARENT_ARTIFACT / "summary.json"),
            "role": CONTRACT["parent_metadata_roles"]["summary.json"],
            "classification": parent_summary["classification"],
            "one_Q_execution_manifest_authorized": parent_summary[
                "one_Q_execution_manifest_authorized"
            ],
        },
        "effective_authorization": (
            "definitions_only_bounded_one_Q_continuation_cost_manifest"
        ),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(ARTIFACT_DIRECTORY / "execution_manifest.json", CONTRACT)
    _write(ARTIFACT_DIRECTORY / "parent_authorization.json", authorization)
    _write(ARTIFACT_DIRECTORY / "seed_inventory.json", inventory)
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "python": sys.version,
            "platform": platform.platform(),
            "source_hashes": _source_hashes(),
            "parent_artifact_hashes": _artifact_hashes(PARENT_ARTIFACT),
            "seed_artifact_hashes": _artifact_hashes(SEED_ARTIFACT),
        },
    )
    names = (
        "execution_manifest.json",
        "parent_authorization.json",
        "provenance.json",
        "seed_inventory.json",
        "summary.json",
    )
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _catalog(ARTIFACT_DIRECTORY, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        raise SystemExit("select --freeze")
    payload = _freeze()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
