#!/usr/bin/env python3
"""Freeze an execution-shaped constrained BDF1/BDF2 history preflight."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24d"
ARTIFACT = (
    "causal_inner_face36_fixed_q_constrained_history_manifest_"
    "wp10c9d6c7c3b5c4f24d"
)
PARENT_ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_second_state_bdf2_preflight_"
    "wp10c9d6c7c3b5c4f24c"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_constrained_history_manifest_"
    "wp10c9d6c7c3b5c4f24d.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_constrained_history_manifest_"
    "wp10c9d6c7c3b5c4f24d.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FACE36_FIXED_Q_CONSTRAINED_HISTORY_MANIFEST_"
    "WP10C9D6C7C3B5C4F24D_2026-08-14.md"
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
        "fixed_Q_second_state_Jacobian_and_exact_BDF2_roots_passed_"
        "but_synthetic_history_limit_orders_failed"
    )
    if (
        summary["classification"] != expected
        or summary["passed"]
        or not summary["second_state_derivative_certified"]
        or not summary["exact_constrained_BDF2_roots_certified"]
        or summary["synthetic_history_limit_orders_certified"]
        or summary["one_Q_execution_manifest_authorized"]
        or summary["authorized_next"]
        != "definitions_only_constrained_BDF_startup_history_preflight"
    ):
        raise RuntimeError("c4f24d authorization changed")
    return summary


def _manifest(parent: dict) -> dict:
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_constrained_BDF1_startup_BDF2_history_manifest_frozen_"
            "execution_preflight_authorized"
        ),
        "definitions_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "inherited_certificates": {
            "Q3_reaction_and_continuous_KKT": True,
            "complete_state_dependent_Jacobian_at_20ms": True,
            "complete_state_dependent_Jacobian_at_16ms": True,
            "exact_constrained_backward_Euler_limit_at_20ms": True,
            "exact_equal_Q_BDF2_roots_at_16ms": True,
            "synthetic_projected_history_limit_rejected": True,
            "parent_minimum_rate_order": min(
                parent["rate_convergence_orders"]
            ),
            "parent_minimum_multiplier_order": min(
                parent["multiplier_convergence_orders"]
            ),
        },
        "committed_states": [
            {
                "role": "primary",
                "layout": "middle",
                "time_seconds": 0.020,
                "source": "committed_middle_20ms_base_endpoint",
            },
            {
                "role": "held_out",
                "layout": "middle",
                "time_seconds": 0.016,
                "source": "committed_middle_16ms_base_endpoint",
            },
        ],
        "execution_shaped_chain": {
            "timestep_ladder_seconds": [1.0e-7, 5.0e-8, 2.5e-8],
            "fail_fast_order": [
                "primary_coarse_BDF1_then_BDF2",
                "held_out_coarse_BDF1_then_BDF2",
                "primary_middle_and_fine",
                "held_out_middle_and_fine",
                "serialized_replay",
            ],
            "startup": (
                "one_exact_constrained_BDF1_root_from_the_committed_state"
            ),
            "history": (
                "accepted_BDF1_primitive_increment_mapped_storage_increment_"
                "responsive_height_increment_and_timestep"
            ),
            "continuation": (
                "one_exact_equal_step_constrained_BDF2_root_using_only_the_"
                "accepted_BDF1_history"
            ),
            "synthetic_backward_tangent_projection_forbidden": True,
            "manual_primitive_freezing_forbidden": True,
            "Q3_target": "Q3_of_each_committed_start_state",
            "reaction_channel_basis": "frozen_normalized",
            "raw_channel_repeat": "diagnostic_only_if_conditioning_requires",
        },
        "residual_contract": {
            "binding_solver_residual": "increment_primary_complete_BDF",
            "direct_rate_form": "independent_parity_audit_only",
            "maximum_direct_rate_increment_parity_defect": 1.0e-9,
            "state_dependent_reaction_JVP_required": True,
            "constraint_row_scales_frozen_at_step_target": True,
            "one_exact_Jacobian_refresh_then_matrix_free_reaction_corrections": (
                True
            ),
            "dense_exact_refresh_at_every_iteration_forbidden_in_execution": (
                True
            ),
        },
        "binding_gates": {
            "maximum_scaled_residual": 1.0e-10,
            "maximum_Q3_relative_defect": 1.0e-12,
            "maximum_reaction_ledger_relative_defect": 1.0e-12,
            "maximum_constraint_work_ledger_relative_defect": 1.0e-12,
            "maximum_storage_parity_relative_defect": 1.0e-9,
            "maximum_checkpoint_roundtrip_difference": 0.0,
            "maximum_split_replay_difference": 0.0,
            "maximum_reconstruction_factor": 1.0,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_H_over_R": 0.12,
            "incoming_excision_characteristics": 0,
            "minimum_state_rate_convergence_order": 0.9,
            "minimum_reaction_action_convergence_order": 0.9,
        },
        "convergence_observables": {
            "compare_at_chain_endpoint_relative_to_each_start": True,
            "state_rate": "complete_BDF_weighted_scaled_rate",
            "reaction_action": (
                "physical_scaled_B_Q_times_multiplier_not_multiplier_"
                "coordinates_alone"
            ),
            "face36_rate": "certified_exterior_partition_directional_rate",
            "multiplier_coordinates": "reported_nonbinding_conditioning_audit",
            "require_two_adjacent_orders": True,
        },
        "durability": {
            "serialize_after_BDF1_and_after_BDF2": True,
            "replay_BDF2_from_serialized_BDF1_history": True,
            "compare_primitive_mapped_height_multiplier_and_Q3_arrays": True,
            "bitwise_required": True,
            "hash_execution_tree_and_all_solver_dependencies": True,
        },
        "decision_tree": {
            "any_primary_or_held_out_method_gate_fails": (
                "stop_and_localize_startup_history_reaction_or_storage"
            ),
            "roots_pass_but_convergence_fails": (
                "do_not_authorize_one_Q_manifest_audit_history_or_conditioning"
            ),
            "all_gates_and_replay_pass": (
                "authorize_fresh_definitions_only_one_Q_execution_manifest"
            ),
        },
        "cost_contract": {
            "reuse_committed_20ms_BE_evidence_when_hashes_match": True,
            "one_exact_matrix_refresh_per_root": True,
            "fail_fast_before_refined_rungs": True,
            "estimated_wall_hours": [1.5, 4.0],
            "no_long_trajectory": True,
        },
        "hard_stops": [
            "do_not_tune_or_reuse_the_rejected_synthetic_history_projection",
            "do_not_relax_the_1e-10_residual_or_1e-12_Q3_gates",
            "do_not_authorize_a_fixed_Q_microburst_from_exact_roots_alone",
            "do_not_run_50ms_or_fine_or_reduced_slow_evolution",
            "do_not_use_raw_face48_as_the_slow_exchange",
        ],
        "constrained_history_execution_preflight_authorized": True,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "one_Q_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f24e_exact_constrained_BDF1_startup_"
            "BDF2_history_preflight"
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
    parent = _parent()
    manifest = _manifest(parent)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "inherited_second_state_derivative_certified": True,
        "inherited_exact_constrained_BDF2_roots_certified": True,
        "synthetic_history_limit_rejection_preserved": True,
        "constrained_history_execution_preflight_authorized": True,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "one_Q_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
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
        },
    )
    _write(CANONICAL_DIRECTORY / "execution_manifest.json", manifest)
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    source_hashes = {
        THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
        THIS_TEST: _sha(ROOT / THIS_TEST) if (ROOT / THIS_TEST).exists() else None,
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
        "execution_manifest.json",
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
