#!/usr/bin/env python3
"""Freeze the corrected execution contract for the fixed-Q history preflight."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24d1"
ARTIFACT = (
    "causal_inner_face36_fixed_q_constrained_history_contract_correction_"
    "wp10c9d6c7c3b5c4f24d1"
)
PARENT_ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_constrained_history_manifest_"
    "wp10c9d6c7c3b5c4f24d"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_constrained_history_contract_"
    "correction_wp10c9d6c7c3b5c4f24d1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_constrained_history_contract_"
    "correction_wp10c9d6c7c3b5c4f24d1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FACE36_FIXED_Q_CONSTRAINED_HISTORY_CONTRACT_"
    "CORRECTION_WP10C9D6C7C3B5C4F24D1_2026-08-15.md"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
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


def _parent() -> dict:
    summary = _read(PARENT_ARTIFACT / "summary.json")
    expected = (
        "fixed_Q_constrained_BDF1_startup_BDF2_history_manifest_frozen_"
        "execution_preflight_authorized"
    )
    if (
        summary["classification"] != expected
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["constrained_history_execution_preflight_authorized"]
        or summary["one_Q_execution_manifest_authorized"]
        or summary["fixed_Q_micro_solver_authorized"]
        or summary["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("c4f24d contract changed")
    return summary


def _contract() -> dict:
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_constrained_history_contract_corrected_"
            "implementation_preflight_authorized"
        ),
        "definitions_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "supersedes_execution_contract_only": (
            "WP10c9d6c7c3b5c4f24d"
        ),
        "preserved_scientific_evidence": {
            "fixed_Q_Jacobian_repair": True,
            "exact_direct_rate_BDF_roots": True,
            "synthetic_projected_history_rejection": True,
            "Q3_reaction_and_continuous_KKT": True,
        },
        "binding_temporal_contract": {
            "nonlinear_root": "increment_primary_complete_BDF_only",
            "binding_solver_passes_direct_rate": False,
            "direct_rate_evaluation": "post_root_parity_audit_only",
            "direct_rate_uses_current_interval_rate_not_BDF_weighted_rate": True,
            "maximum_direct_rate_increment_parity_defect": 1.0e-9,
            "old_direct_rate_roots_may_seed_but_not_certify_new_execution": True,
        },
        "reaction_contract": {
            "binding_channel_basis": "frozen_normalized",
            "raw_channel_repeat": "diagnostic_only",
            "state_normalized_channels": "audit_only_not_nonlinear_kernel",
            "replace_explicit_inverse_with_stable_solve": True,
            "required_numerical_rank": 3,
            "maximum_raw_Schur_condition_number": 1.0e8,
            "maximum_normalization_identity_defect": 1.0e-12,
            "report_singular_values_rank_condition_and_action_sensitivity": True,
        },
        "solver_contract": {
            "initial_matrix": (
                "one_complete_bordered_Jacobian_at_the_initial_unknown"
            ),
            "subsequent_updates": "dense_rank_one_Broyden_secant_updates",
            "matrix_free_claim": False,
            "maximum_complete_Jacobian_assemblies_per_binding_root": 1,
            "extra_exact_refresh": "diagnostic_only_cannot_convert_failure_to_pass",
            "record_assembly_Broyden_and_linear_solve_counts": True,
        },
        "acceptance_contract": {
            "single_fail_closed_acceptance_record": True,
            "BDF2_requires_accepted_BDF1_history": True,
            "rejected_candidate_may_be_saved_but_not_advanced": True,
            "maximum_scaled_residual": 1.0e-10,
            "maximum_Q3_relative_defect": 1.0e-12,
            "maximum_reaction_channel_ledger_relative_defect": 1.0e-12,
            "maximum_multiplier_weighted_action_ledger_relative_defect": 1.0e-12,
            "maximum_storage_parity_relative_defect": 1.0e-9,
            "minimum_path_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_path_reconstruction_factor_sanity": 1.0 + 1.0e-12,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "incoming_excision_characteristics": 0,
            "serialize_all_failure_reasons": True,
        },
        "constraint_diagnostics": {
            "reaction_channel_ledger": (
                "unweighted_cell_sum_vs_channel_matrix_identity"
            ),
            "constraint_action_ledger": (
                "multiplier_weighted_cell_sum_vs_channel_action"
            ),
            "endpoint_Q3_constraint": "binding",
            "continuous_DQ3_tangency": "reported_separately",
            "finite_BDF_DQ3_times_state_rate": (
                "diagnostic_not_subject_to_the_1e-12_ledger_gate"
            ),
        },
        "restart_contract": {
            "primitive_and_complete_monolithic_history": True,
            "Q3_target_and_constraint_scales": True,
            "multiplier_predictor": True,
            "reaction_basis_and_transform_policy": True,
            "elapsed_time_order_and_timestep_history": True,
            "source_configuration_and_environment_hashes": True,
            "recompute_state_local_reaction_after_reload": True,
            "bitwise_BDF2_replay": True,
        },
        "execution_contract": {
            "states_seconds": [0.020, 0.016],
            "timestep_ladder_seconds": [1.0e-7, 5.0e-8, 2.5e-8],
            "fail_fast_order": [
                "both_states_coarse",
                "both_states_middle",
                "both_states_fine",
                "serialized_replay",
            ],
            "one_constrained_BDF1_then_one_equal_step_BDF2": True,
            "synthetic_or_projected_history_forbidden": True,
            "minimum_state_rate_convergence_order": 0.9,
            "minimum_state_space_reaction_action_convergence_order": 0.9,
            "multiplier_coordinate_order": "diagnostic_only",
            "face36_rate": "reported_without_a_new_post_hoc_gate",
        },
        "provenance_contract": {
            "execute_from_clean_worktree_at_exact_committed_SHA": True,
            "unrelated_user_untracked_files_must_not_be_modified": True,
            "record_commit_tree_command_environment_and_dependency_hashes": True,
            "record_BLAS_identity_and_thread_configuration": True,
            "canonical_SHA256SUMS_required": True,
        },
        "decision_tree": {
            "all_gates_pass": (
                "authorize_fresh_definitions_only_bounded_one_Q_manifest"
            ),
            "binding_root_fails": "stop_and_localize_solver_or_conditioning",
            "temporal_parity_fails": "stop_and_localize_storage_history",
            "reaction_action_order_fails": (
                "stop_and_localize_reaction_conditioning_or_normalization"
            ),
            "replay_fails": "stop_and_localize_restart_or_determinism",
            "finest_pair_hits_numerical_floor": (
                "classify_inconclusive_and_freeze_new_ladder_prospectively"
            ),
        },
        "hard_stops": [
            "do_not_execute_f24e_with_the_uncorrected_direct_rate_binding_solver",
            "do_not_rewrite_prior_canonical_results",
            "do_not_relax_residual_Q3_or_order_gates",
            "do_not_use_raw_face48_as_slow_exchange",
            "do_not_start_fixed_Q_microburst_50ms_or_reduced_evolution",
        ],
        "implementation_preflight_authorized": True,
        "physical_history_execution_authorized": False,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f24e0_increment_primary_solver_restart_"
            "acceptance_hardening"
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
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
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


def main() -> None:
    _parent()
    contract = _contract()
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": contract["classification"],
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "prior_scientific_evidence_preserved": True,
        "uncorrected_f24e_execution_blocked": True,
        "implementation_preflight_authorized": True,
        "physical_history_execution_authorized": False,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": contract["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(
        CANONICAL_DIRECTORY / "config.json",
        {
            "schema_version": 1,
            "work_package": WORK_PACKAGE,
            "states_seconds": [0.020, 0.016],
            "timestep_ladder_seconds": [1.0e-7, 5.0e-8, 2.5e-8],
            "shared_exchange_parent_face": 36,
            "raw_Schur_condition_gate": 1.0e8,
        },
    )
    _write(CANONICAL_DIRECTORY / "corrected_contract.json", contract)
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    source_hashes = {
        THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
        THIS_TEST: _sha(ROOT / THIS_TEST),
        REPORT_RELATIVE: _sha(ROOT / REPORT_RELATIVE),
    }
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "source_parent_commit": _git("rev-parse", "HEAD"),
            "source_parent_tree": _git("rev-parse", "HEAD^{tree}"),
            "parent_summary_sha256": _sha(PARENT_ARTIFACT / "summary.json"),
            "source_hashes": source_hashes,
        },
    )
    files = (
        "config.json",
        "corrected_contract.json",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in files
        ),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
