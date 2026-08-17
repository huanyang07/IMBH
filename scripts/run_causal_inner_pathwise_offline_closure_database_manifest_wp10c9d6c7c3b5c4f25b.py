#!/usr/bin/env python3
"""Freeze the pathwise offline closure-database and pilot contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_reduced_cycle_identifiability_wp10c9d6c7c3b5c4f25a as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25b"
CLASSIFICATION = (
    "pathwise_offline_closure_database_manifest_frozen_"
    "single_anchor_descriptor_pilot_authorized"
)
PARENT_PACKAGE_COMMIT = "a7e0beb3099075cb89c70c9385d47f7d0bd1bec1"
PARENT_PACKAGE_PARENT = "1c070c5c2ab3bcdaaae411fe0c4dfe87752dc4a4"
PARENT_PACKAGE_TREE = "08aa4feb2eee1c59c7a6d81fdaf70986a0950153"

ARTIFACT = "causal_inner_pathwise_offline_closure_database_manifest_wp10c9d6c7c3b5c4f25b"
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_pathwise_offline_closure_database_manifest_"
    "wp10c9d6c7c3b5c4f25b.py"
)
THIS_TEST = (
    "tests/test_causal_inner_pathwise_offline_closure_database_manifest_"
    "wp10c9d6c7c3b5c4f25b.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_pathwise_closure_descriptor_pilot_"
    "wp10c9d6c7c3b5c4f25c.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_pathwise_closure_descriptor_pilot_"
    "wp10c9d6c7c3b5c4f25c.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PATHWISE_OFFLINE_CLOSURE_DATABASE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25B_2026-08-17.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PILOT_SEED_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)
PILOT_SEED_FILES = (
    "summary.json",
    "checkpoint_warm_3.json",
    "checkpoint_warm_3.npz",
    "metrics_warm_3.json",
)

BRANCH_ORDER = ("cold", "transition_up", "hot", "transition_down")
INITIAL_TRAINING_ANCHORS = 12
SEALED_HELDOUT_ANCHORS = 6
INITIAL_ANCHORS = INITIAL_TRAINING_ANCHORS + SEALED_HELDOUT_ANCHORS
MAXIMUM_ADAPTIVE_TRAINING_ANCHORS = 12
MAXIMUM_TOTAL_MIDDLE_ANCHORS = INITIAL_ANCHORS + MAXIMUM_ADAPTIVE_TRAINING_ANCHORS
FINE_VALIDATION_IDS = ("C04", "TU01", "H04", "TD01")
MEMORY_ORDERS = (0, 2, 4, 6)
FREQUENCY_COUNT = 32
FIDUCIAL_CYCLE_SECONDS = 6.7 * 86_400.0
CERTIFIED_FAST_TIMESTEP_SECONDS = 1.0e-7
MAXIMUM_TOTAL_EXECUTION_WALL_HOURS = 72.0


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = _sha(directory / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = actual
    return recorded


def _validate_parent() -> tuple[dict, dict]:
    if _git("rev-parse", PARENT_PACKAGE_COMMIT) != PARENT_PACKAGE_COMMIT:
        raise RuntimeError("parent identifiability package commit changed")
    if _git("rev-parse", f"{PARENT_PACKAGE_COMMIT}^") != PARENT_PACKAGE_PARENT:
        raise RuntimeError("parent identifiability package parent changed")
    if _git("rev-parse", f"{PARENT_PACKAGE_COMMIT}^{{tree}}") != PARENT_PACKAGE_TREE:
        raise RuntimeError("parent identifiability package tree changed")
    hashes = _checksums(parent.ARTIFACT_DIRECTORY)
    summary = _read(parent.ARTIFACT_DIRECTORY / "summary.json")
    decision = _read(parent.ARTIFACT_DIRECTORY / "decision.json")
    if (
        not summary["passed"]
        or not summary["offline_closure_database_manifest_authorized"]
        or summary["online_reduced_solver_implementation_authorized"]
        or summary["authorized_next"]
        != "definitions_only_pathwise_offline_closure_database_manifest"
        or decision["selected_architecture"]
        != "cellwise_Q5_FV_plus_a2_finite_memory_hybrid"
    ):
        raise RuntimeError("parent closure-database authorization changed")
    return summary, {"package_hashes": hashes, "decision": decision}


def _slot(anchor_id: str, branch: str, fraction: float, role: str) -> dict:
    return {
        "anchor_id": anchor_id,
        "branch": branch,
        "normalized_branch_arclength": fraction,
        "role": role,
        "middle_layout_query": True,
        "fine_layout_query": anchor_id in FINE_VALIDATION_IDS,
        "sealed_until_training_and_memory_order_are_frozen": role == "heldout",
    }


def _anchor_schedule() -> dict:
    slots = [
        _slot("C00", "cold", 0.0, "training"),
        _slot("C01", "cold", 0.2, "heldout"),
        _slot("C02", "cold", 0.4, "training"),
        _slot("C03", "cold", 0.6, "training"),
        _slot("C04", "cold", 0.8, "heldout"),
        _slot("C05", "cold", 1.0, "training"),
        _slot("TU00", "transition_up", 0.0, "training"),
        _slot("TU01", "transition_up", 0.5, "heldout"),
        _slot("TU02", "transition_up", 1.0, "training"),
        _slot("H00", "hot", 0.0, "training"),
        _slot("H01", "hot", 0.2, "heldout"),
        _slot("H02", "hot", 0.4, "training"),
        _slot("H03", "hot", 0.6, "training"),
        _slot("H04", "hot", 0.8, "heldout"),
        _slot("H05", "hot", 1.0, "training"),
        _slot("TD00", "transition_down", 0.0, "training"),
        _slot("TD01", "transition_down", 0.5, "heldout"),
        _slot("TD02", "transition_down", 1.0, "training"),
    ]
    training = [slot["anchor_id"] for slot in slots if slot["role"] == "training"]
    heldout = [slot["anchor_id"] for slot in slots if slot["role"] == "heldout"]
    if len(slots) != INITIAL_ANCHORS:
        raise RuntimeError("initial anchor count changed")
    if len(training) != INITIAL_TRAINING_ANCHORS or len(heldout) != SEALED_HELDOUT_ANCHORS:
        raise RuntimeError("training/heldout split changed")
    if set(FINE_VALIDATION_IDS) - set(heldout):
        raise RuntimeError("fine validation must be a subset of sealed heldout anchors")
    return {
        "path_is_one_dimensional_branch_arclength_not_a_tensor_product_grid": True,
        "branch_order": BRANCH_ORDER,
        "initial_slots": slots,
        "initial_training_ids": training,
        "sealed_heldout_ids": heldout,
        "fine_validation_ids": FINE_VALIDATION_IDS,
        "initial_anchor_count": len(slots),
        "maximum_adaptive_training_anchors": MAXIMUM_ADAPTIVE_TRAINING_ANCHORS,
        "maximum_total_middle_anchors": MAXIMUM_TOTAL_MIDDLE_ANCHORS,
        "adaptive_rule": {
            "heldout_data_may_not_select_new_anchors": True,
            "indicator": "training_only_leave_one_interval_out_flux_and_transfer_error",
            "new_location": "midpoint_of_highest_indicator_same_branch_interval",
            "deterministic_tie_break": "branch_order_then_lower_arclength",
            "maximum_new_anchors_per_branch": 3,
            "stop_when_all_training_indicators_pass_or_budget_is_exhausted": True,
        },
        "physical_anchor_construction": {
            "cold_and_hot": (
                "constraint_consistent_pseudo_arclength_roots_in_total_inner_mass_"
                "with_all_other_slow_controls_recorded_not_silently_frozen"
            ),
            "transitions": (
                "event_bracketed_states_from_the_shortest_prospectively_authorized_"
                "truth_snippet_not_interpolation_between_branch_roots"
            ),
            "branch_existence_is_assumed": False,
            "one_zone_thresholds_define_truth_anchors": False,
            "failure_to_find_two_branches_and_two_transitions": (
                "database_architecture_not_identified_stop_without_cycle_claim"
            ),
        },
    }


def _frequency_grid() -> dict:
    omega_min = 2.0 * math.pi / FIDUCIAL_CYCLE_SECONDS
    omega_max = math.pi / CERTIFIED_FAST_TIMESTEP_SECONDS
    ratio = (omega_max / omega_min) ** (1.0 / (FREQUENCY_COUNT - 1))
    values = [omega_min * ratio**index for index in range(FREQUENCY_COUNT)]
    return {
        "count": FREQUENCY_COUNT,
        "angular_frequency_min_per_second": omega_min,
        "angular_frequency_max_per_second": omega_max,
        "spacing": "logarithmic_in_angular_frequency",
        "values_per_second": values,
        "includes_exact_DC_evaluation_separately": True,
    }


def _database_contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "selected_architecture": "cellwise_Q5_FV_plus_a2_finite_memory_hybrid",
        "purpose": (
            "identify_a_trajectory_tube_closure_from_offline_truth_without_"
            "placing_the_truth_solver_in_the_online_cycle"
        ),
        "identification_scope": {
            "pathwise_not_global_state_space": True,
            "global_tensor_product_Q_grid_forbidden": True,
            "local_validity_object": "branchwise_trajectory_tube_with_linear_transverse_response",
            "middle_layout_is_training_truth": True,
            "fine_layout_is_sparse_validation_truth_only": True,
            "online_truth_calls": 0,
        },
        "resolved_coordinates": {
            "primary_radial_cells": 16,
            "cellwise_storage": (
                "mapped_mass",
                "mapped_angular_momentum",
                "mapped_killing_energy",
                "column_thermal_content_candidate",
                "relaxing_stress_storage_candidate",
            ),
            "explicit_cross_grid_stable_amplitudes": 2,
            "candidate_memory_orders": MEMORY_ORDERS,
            "branch_labels": BRANCH_ORDER,
            "restriction_must_preserve_cell_integrated_M_J_E_exactly": True,
            "lifting_must_satisfy_RL_identity_and_constraints": True,
        },
        "truth_record_per_anchor": {
            "required_state_fields": (
                "primitive_state",
                "mapped_storage",
                "responsive_height_history",
                "Q_target_and_constraint_scales",
                "branch_and_arclength",
            ),
            "required_quasi_steady_outputs": (
                "single_valued_M_J_E_flux_at_every_coarse_face",
                "face36_exterior_partition",
                "cell_sources_and_work",
                "two_mode_coordinates_and_rates",
                "physical_reaction_action",
            ),
            "required_linear_objects": (
                "continuous_time_descriptor_E_and_A",
                "constraint_border_and_reaction_action_derivative",
                "conservative_restriction_and_constraint_compatible_lifting",
                "resolved_to_flux_transfer_G_of_s",
                "two_explicit_mode_projectors",
            ),
            "frequency_grid": _frequency_grid(),
            "raw_pointwise_horizon_flux_forbidden": True,
        },
        "descriptor_reduction": {
            "full_linearization": "E_dxdt=A_dx+B_du_with_constraint_border",
            "state_split": "dx=L_dU+Z_dz_with_RL=I_and_RZ=0",
            "unresolved_transfer": (
                "G(s)=D+C_z*(s*E_z-A_z)^(-1)*(A_z*L-s*E_z*L)"
            ),
            "two_stable_modes_are_removed_from_the_kernel_and_retained_explicitly": True,
            "remaining_transfer_is_fit_with_shared_stable_poles": True,
            "DC_quasi_steady_map_is_not_refit_by_the_memory_kernel": True,
        },
        "memory_selection": {
            "orders_tested": MEMORY_ORDERS,
            "selection_rule": "smallest_order_passing_every_training_CV_and_sealed_heldout_gate",
            "poles_must_have_strictly_negative_real_parts": True,
            "complex_poles_and_residues_must_occur_in_conjugate_pairs": True,
            "passivity_or_declared_supply_rate_dissipation_identity_required": True,
            "post_heldout_order_change_forbidden": True,
        },
        "binding_validation_gates": {
            "truth_root_complete_residual_max": 1.0e-10,
            "complete_JVP_relative_defect_max": 1.0e-7,
            "restriction_lifting_identity_max": 5.0e-12,
            "M_J_E_telescope_relative_defect_max": 5.0e-12,
            "frequency_solve_relative_residual_max": 1.0e-10,
            "quasi_steady_face_flux_normalized_error_max": 0.15,
            "significant_transfer_direction_error_max": 0.25,
            "leading_two_projector_cosine_min": 0.95,
            "fine_middle_significant_transfer_error_max": 0.25,
            "unstable_fitted_pole_count_max": 0,
            "online_conservation_regression_must_be_bitwise": True,
        },
        "claim_boundary": {
            "database_pass_may_authorize": (
                "online_reduced_solver_implementation_manifest",
                "committed_short_time_replay_manifest",
            ),
            "database_pass_may_not_authorize": (
                "predictive_QPE_cycle",
                "physical_hot_cold_cycle_claim",
                "reduced_slow_evolution",
            ),
            "exploratory_one_zone_switches_are_not_truth_labels": True,
        },
        "hard_stops": (
            "do_not_open_heldout_anchors_before_training_and_memory_order_are_frozen",
            "do_not_add_anchors_using_heldout_error",
            "do_not_call_an_unclassified_20ms_seed_a_cold_or_hot_state",
            "do_not_infer_transition_states_by_interpolating_steady_branches",
            "do_not_run_a_global_tensor_product_parameter_sweep",
            "do_not_put_fixed_Q_or_HMM_truth_calls_in_the_online_model",
            "do_not_claim_a_predictive_cycle_from_the_database_alone",
        ),
    }


def _pilot_contract() -> dict:
    seed_hashes = {
        str((PILOT_SEED_DIRECTORY / name).relative_to(ROOT)): _sha(PILOT_SEED_DIRECTORY / name)
        for name in PILOT_SEED_FILES
    }
    return {
        "pilot_id": "P00_primary_20ms_descriptor_schema",
        "purpose": "test_descriptor_and_database_extractability_before_any_anchor_campaign",
        "seed_role": "unclassified_schema_seed_not_training_not_heldout",
        "seed_branch_label": "unclassified",
        "seed_hashes": seed_hashes,
        "allowed_new_nonlinear_roots": 0,
        "allowed_exact_continuous_descriptor_assemblies": 1,
        "allowed_frequency_points": FREQUENCY_COUNT + 1,
        "allowed_short_truth_burst_steps": 0,
        "allowed_fine_layout_queries": 0,
        "maximum_wall_hours": 6.0,
        "must_extract": (
            "conservative_16_cell_restriction",
            "constraint_compatible_lifting",
            "continuous_time_descriptor_not_a_BDF_matrix",
            "two_explicit_mode_projectors",
            "all_coarse_face_M_J_E_output_rows",
            "DC_and_log_frequency_transfer_samples",
        ),
        "pass_requires": {
            "seed_checkpoint_hash_and_acceptance_valid": True,
            "no_new_root_or_propagated_state": True,
            "descriptor_complete_JVP_relative_defect_max": 1.0e-7,
            "restriction_lifting_identity_max": 5.0e-12,
            "M_J_E_telescope_relative_defect_max": 5.0e-12,
            "frequency_solve_relative_residual_max": 1.0e-10,
            "transfer_conjugate_symmetry_relative_defect_max": 1.0e-10,
            "database_roundtrip_bitwise": True,
        },
        "classifications": {
            "pass": "single_anchor_descriptor_schema_passed_first_training_batch_manifest_authorized",
            "fail": "single_anchor_descriptor_schema_failed_database_campaign_blocked",
            "inconclusive": "single_anchor_descriptor_schema_inconclusive_diagnostic_only",
        },
        "automatic_next_on_pass": (
            "definitions_only_first_training_anchor_batch_manifest_not_the_full_campaign"
        ),
        "full_campaign_truth_budget": {
            "maximum_middle_anchors": MAXIMUM_TOTAL_MIDDLE_ANCHORS,
            "maximum_fine_validation_anchors": len(FINE_VALIDATION_IDS),
            "maximum_optional_transition_burst_anchors": 2,
            "maximum_optional_transition_burst_steps_per_anchor": 8,
            "maximum_total_execution_wall_hours": MAXIMUM_TOTAL_EXECUTION_WALL_HOURS,
            "stop_on_first_binding_failure": True,
        },
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": _sha(path),
                "scientific_status": "PROSPECTIVE",
            })
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": PARENT_PACKAGE_COMMIT,
        "latest_work_package": WORK_PACKAGE,
    })
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parent_summary, parent_lock = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("closure-database manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("closure-database manifest is already frozen")
    schedule = _anchor_schedule()
    contract = _database_contract()
    pilot = _pilot_contract()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "initial_training_anchors": INITIAL_TRAINING_ANCHORS,
        "sealed_heldout_anchors": SEALED_HELDOUT_ANCHORS,
        "maximum_total_middle_anchors": MAXIMUM_TOTAL_MIDDLE_ANCHORS,
        "single_anchor_descriptor_pilot_authorized": True,
        "full_anchor_campaign_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "parent_classification_preserved": parent_summary["classification"],
        "authorized_next": "WP10c9d6c7c3b5c4f25c_single_anchor_descriptor_pilot",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "closure_database_contract.json", contract)
    _write(ARTIFACT_DIRECTORY / "anchor_schedule.json", schedule)
    _write(ARTIFACT_DIRECTORY / "pilot_contract.json", pilot)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", {
        "parent_package_commit": PARENT_PACKAGE_COMMIT,
        "parent_package_parent": PARENT_PACKAGE_PARENT,
        "parent_package_tree": PARENT_PACKAGE_TREE,
        **parent_lock,
    })
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "provenance.json", {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "PROSPECTIVE",
        "definition_commit": _git("rev-parse", "HEAD"),
        "definition_tree": _git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean_at_start": True,
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "authorized_next_runner": NEXT_RUNNER,
        "authorized_next_test": NEXT_TEST,
        "report": REPORT_RELATIVE,
        "source_hashes": {
            THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
            THIS_TEST: _sha(ROOT / THIS_TEST),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    })
    REPORT_PATH.write_text(
        "\n".join((
            "# Pathwise offline closure-database manifest WP10c9d6c7c3b5c4f25b",
            "",
            "## Classification",
            "",
            f"`{CLASSIFICATION}`",
            "",
            "This package freezes the offline data design; it runs no new root, trajectory, or tangent propagation. The database is a branchwise trajectory tube, not a global tensor-product grid.",
            "",
            "The initial design has `12` training and `6` sealed held-out middle-layout anchors distributed across cold, up-transition, hot, and down-transition segments. Training-only error indicators may add at most `12` anchors, for a hard maximum of `30`. Held-out results may never choose new anchors. Four held-out anchors are prospectively selected for sparse fine-layout validation.",
            "",
            "At each valid anchor the truth record must contain conservative 16-cell storage and face fluxes, the two stable explicit modes, an exact continuous-time constrained descriptor, and the resolved-to-flux transfer function at DC plus 32 logarithmic frequencies. Memory orders `0/2/4/6` are compared; the smallest order passing every frozen gate is selected. Stable poles, conjugate pairing, exact M/J/E telescoping, and a declared dissipation identity are binding.",
            "",
            "Branch existence is not assumed. The phenomenological one-zone thresholds do not label truth states, and the accepted 20 ms seed is explicitly unclassified. Failure to find two physical branches and two event-bracketed transitions stops the predictive-cycle route.",
            "",
            "Only one nonpropagating descriptor-schema pilot is authorized next. It reuses the hash-locked accepted primary checkpoint, permits no new nonlinear root, one exact continuous descriptor assembly, DC plus 32 frequency evaluations, no burst, and no fine-grid query. A pass may authorize only a definitions-only first training-batch manifest.",
            "",
            "No full anchor campaign, online reduced solver, predictive cycle, or reduced slow evolution is authorized.",
            "",
        )),
        encoding="utf-8",
    )
    names = (
        "anchor_schedule.json",
        "closure_database_contract.json",
        "parent_lock.json",
        "pilot_contract.json",
        "provenance.json",
        "summary.json",
    )
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
