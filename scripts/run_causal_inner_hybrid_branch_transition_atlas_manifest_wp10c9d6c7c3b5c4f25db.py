#!/usr/bin/env python3
"""Freeze separate slow-branch and fast-transition atlas contracts."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_local_slaving_transition_diagnosis_wp10c9d6c7c3b5c4f25da as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25db"
PARENT_COMMIT = "d09b51dc6ad3e967453cdf48f233cd0199744dbd"
PARENT_PARENT = "ff4fc9917a067069ef2613a6499944129b919d3d"
PARENT_TREE = "d7131a6f949427509d42c6f0f9f55c6698c4f742"

CLASSIFICATION = (
    "hybrid_branch_transition_atlas_contract_frozen_"
    "candidate_geometry_preflight_authorized"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dc"

BRANCHES = ("cold", "hot")
TRANSITIONS = ("cold_to_hot", "hot_to_cold")
MEMORY_ORDERS = (0, 2, 4, 6)
BRANCH_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BRANCH_TRAINING_INDICES = (0, 2, 3, 5)
BRANCH_HELDOUT_INDICES = (1, 4)
TRANSITION_TRAINING_QUANTILES = (0.2, 0.5, 0.8)
TRANSITION_HELDOUT_QUANTILES = (0.65,)

ARTIFACT = (
    "causal_inner_hybrid_branch_transition_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25db"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_hybrid_branch_transition_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25db.py"
)
THIS_TEST = (
    "tests/test_causal_inner_hybrid_branch_transition_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25db.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HYBRID_BRANCH_TRANSITION_ATLAS_"
    "MANIFEST_WP10C9D6C7C3B5C4F25DB_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
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


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("hybrid atlas parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("hybrid atlas parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("hybrid atlas parent tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    architecture = _read(parent.CANONICAL_DIRECTORY / "revised_architecture.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["classification"] != parent.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["forward_patch_is_transition_layer_seed"]
        or summary["forward_patch_is_slow_graph"]
        or architecture["next_definitions_only_package"]["work_package"]
        != WORK_PACKAGE
    ):
        raise RuntimeError("hybrid atlas authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"diagnosis source changed: {relative}")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("hybrid atlas manifest requires a clean tracked tree")
    return {"summary": summary, "architecture": architecture, "hashes": hashes}


def _branch_slots() -> list[dict]:
    slots = []
    for branch_code, branch in (("C", "cold"), ("H", "hot")):
        for index, fraction in enumerate(BRANCH_FRACTIONS):
            role = "training" if index in BRANCH_TRAINING_INDICES else "heldout"
            slots.append(
                {
                    "anchor_id": f"{branch_code}{index:02d}",
                    "branch": branch,
                    "normalized_branch_arclength": fraction,
                    "role": role,
                    "middle_layout": True,
                    "fine_layout": role == "heldout" and index == 4,
                    "sealed_until_memory_order_frozen": role == "heldout",
                }
            )
    return slots


def _transition_slots() -> list[dict]:
    slots = []
    for prefix, direction in (("U", "cold_to_hot"), ("D", "hot_to_cold")):
        for index, quantile in enumerate(TRANSITION_TRAINING_QUANTILES):
            slots.append(
                {
                    "transition_id": f"{prefix}T{index:02d}",
                    "direction": direction,
                    "event_surface_quantile": quantile,
                    "role": "training",
                    "middle_layout": True,
                    "fine_layout": False,
                    "sealed": False,
                }
            )
        for index, quantile in enumerate(TRANSITION_HELDOUT_QUANTILES):
            slots.append(
                {
                    "transition_id": f"{prefix}V{index:02d}",
                    "direction": direction,
                    "event_surface_quantile": quantile,
                    "role": "heldout",
                    "middle_layout": True,
                    "fine_layout": True,
                    "sealed": True,
                }
            )
    return slots


def _branch_contract() -> dict:
    slots = _branch_slots()
    training = [item["anchor_id"] for item in slots if item["role"] == "training"]
    heldout = [item["anchor_id"] for item in slots if item["role"] == "heldout"]
    return {
        "schema_version": SCHEMA_VERSION,
        "object": "quasi_steady_slow_branch_closure",
        "branches": BRANCHES,
        "online_state": "U80_Q5_plus_a2_plus_selected_stable_memory_mr",
        "anchor_schedule": slots,
        "training_ids": training,
        "sealed_heldout_ids": heldout,
        "branch_state_definition": {
            "constraints": "R82_y_equals_X82",
            "fast_root": "Z_fast_transpose_F_atlas_y_equals_zero",
            "stability": "all_unresolved_fast_eigenvalues_strictly_negative",
            "minimum_spectral_gap_ratio": 10.0,
            "maximum_relative_invariance_defect": 0.10,
            "construction": (
                "constraint_consistent_pseudo_arclength_continuation_with_no_"
                "post_root_projection"
            ),
            "branch_existence_is_not_assumed": True,
        },
        "local_conservative_closure": {
            "fit_object": "one_single_valued_face_flux_per_radial_face",
            "pooled_locality": (
                "shared_radius_conditioned_stencil_law_across_16_cells_"
                "rather_than_one_global_80_dimensional_polynomial"
            ),
            "interior_stencil": "neighboring_Q5_cells_plus_a2_plus_memory",
            "boundary_faces": "separate_certified_inner_and_outer_boundary_maps",
            "cell_derivative_regression_forbidden": True,
            "independent_left_and_right_face_predictions_forbidden": True,
            "global_M_J_E_telescoping_exact_by_construction": True,
        },
        "stable_memory": {
            "orders": MEMORY_ORDERS,
            "model": "stable_passive_shared_pole_rational_Mori_Zwanzig_kernel",
            "selection": "smallest_training_cross_validation_order_passing_all_gates",
            "freeze_before_opening_heldout": True,
            "unstable_or_transition_coordinates_may_not_be_memory": True,
            "if_order_6_fails": "expand_explicit_branch_state_not_memory_order",
        },
        "binding_gates": {
            "complete_truth_root_residual_max": 1.0e-10,
            "restriction_lifting_identity_max": 5.0e-12,
            "global_M_J_E_telescope_relative_defect_max": 5.0e-12,
            "fast_spectral_abscissa_max_per_second": 0.0,
            "spectral_gap_ratio_min": 10.0,
            "slow_graph_invariance_relative_defect_max": 0.10,
            "heldout_face_flux_normalized_error_max": 0.15,
            "heldout_projected_macro_rate_relative_error_max": 0.10,
            "heldout_memory_transfer_relative_error_max": 0.10,
            "fine_middle_significant_flux_error_max": 0.20,
            "stable_poles_required": True,
            "passivity_or_declared_dissipation_required": True,
        },
        "truth_budget": {
            "initial_middle_training_anchors": len(training),
            "sealed_middle_heldout_anchors": len(heldout),
            "fine_heldout_anchors": sum(item["fine_layout"] for item in slots),
            "maximum_adaptive_training_anchors_per_branch": 2,
            "heldout_error_may_not_select_new_anchor": True,
            "stop_on_first_branch_existence_stability_or_training_gate_failure": True,
        },
    }


def _transition_contract() -> dict:
    slots = _transition_slots()
    return {
        "schema_version": SCHEMA_VERSION,
        "object": "fast_event_entry_to_exit_impulse_map",
        "directions": TRANSITIONS,
        "validated_forward_patch_role": "unclassified_transition_geometry_seed",
        "validated_forward_patch_may_not_be_labeled_by_interpolation": True,
        "transition_schedule": slots,
        "entry": {
            "state_source": "accepted_stable_branch_endpoint",
            "surface": "branch_fast_spectral_abscissa_zero_or_fold_discriminant_zero",
            "location": "bracketed_root_in_slow_macro_time",
            "hysteresis_side_recorded": True,
        },
        "exit": {
            "target": "opposite_stable_branch_capture_tube",
            "fast_rate_decay_required": True,
            "branch_distance_and_stability_required": True,
            "maximum_exit_graph_relative_distance": 0.05,
            "failure_to_capture": "transition_map_rejected_no_online_reset",
        },
        "record_per_transition": [
            "entry_and_exit_U80_a2",
            "entry_and_exit_stable_memory_state",
            "event_parameters_and_branch_labels",
            "physical_transition_duration",
            "integrated_single_valued_face_fluxes",
            "integrated_cell_sources_and_constraint_work",
            "Delta_global_M_J_E_ledger",
            "three_internal_active_coordinates_for_offline_diagnosis_only",
        ],
        "online_operator": (
            "T_direction:(X_minus,event_parameters)->"
            "(X_plus,integrated_ledger,new_branch)"
        ),
        "online_transition_ODE_forbidden": True,
        "online_truth_calls": 0,
        "online_fast_microsteps": 0,
        "binding_gates": {
            "complete_truth_residual_max": 1.0e-10,
            "capture_success_fraction": 1.0,
            "heldout_exit_macro_state_relative_error_max": 0.05,
            "heldout_integrated_face_flux_relative_error_max": 0.10,
            "heldout_transition_duration_relative_error_max": 0.20,
            "global_M_J_E_jump_ledger_relative_defect_max": 1.0e-10,
            "fine_middle_transition_impulse_error_max": 0.20,
            "correct_exit_branch_required": True,
            "trust_domain_containment_required": True,
        },
        "truth_budget": {
            "middle_training_transitions": sum(
                item["role"] == "training" for item in slots
            ),
            "sealed_middle_heldout_transitions": sum(
                item["role"] == "heldout" for item in slots
            ),
            "fine_heldout_transitions": sum(item["fine_layout"] for item in slots),
            "first_execution_authorizes_only_one_direction_one_transition_pilot": True,
            "full_schedule_requires_pilot_pass": True,
            "stop_on_first_capture_or_ledger_failure": True,
        },
    }


def _campaign_contract(branch: dict, transition: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "superseded_combined_database_design": {
            "combined_cold_transition_hot_arclength_fit_rejected": True,
            "reason": (
                "the_validated_forward_patch_contains_an_unstable_transition_"
                "direction_and_is_not_a_quasi_steady_branch_graph"
            ),
            "prior_numerical_evidence_modified": False,
        },
        "identification_sequence": [
            "candidate_geometry_preflight_using_existing_artifacts_only",
            "two_branch_existence_and_stability_pilot",
            "branch_training_only_fit_and_memory_order_freeze",
            "open_branch_heldouts_and_sparse_fine_validation",
            "one_direction_one_transition_capture_pilot",
            "transition_training_then_sealed_heldouts",
            "online_hybrid_integrator_implementation_manifest",
            "short_matched_time_replay_before_any_cycle",
        ],
        "fail_fast_dependencies": {
            "transition_training_requires_stable_entry_and_exit_branches": True,
            "heldout_branch_data_open_only_after_memory_order_frozen": True,
            "heldout_transition_data_open_only_after_map_form_frozen": True,
            "online_implementation_requires_both_database_certificates": True,
        },
        "runtime_contract": {
            "fiducial_cycle_seconds": 578880.0,
            "wall_budget_seconds": 259200.0,
            "maximum_online_macrosteps_per_cycle": 100000,
            "minimum_average_macrostep_seconds": 5.7888,
            "maximum_transition_events_per_cycle_for_budgeting": 4,
            "transition_cost_online": "one_interpolated_reset_per_event",
            "truth_and_470_field_online_calls": 0,
            "fast_stability_scale_present_in_online_step_restriction": False,
        },
        "next_package": {
            "work_package": AUTHORIZED_NEXT,
            "classification_on_pass": (
                "hybrid_branch_transition_candidate_geometry_passed_"
                "branch_existence_pilot_manifest_authorized"
            ),
            "new_exact_rate_calls": 0,
            "new_nonlinear_roots": 0,
            "propagated_states": 0,
            "tasks": [
                "inventory_existing_accepted_states_with_exact_U80_coordinates",
                "freeze_unclassified_candidate_branch_pairs_without_assigning_labels",
                "test_macro_path_span_and_physical_guards",
                "freeze_event_surface_observables_independent_of_one_zone_switches",
                "select_one_fail_fast_branch_existence_pilot_before_truth",
            ],
        },
        "authorization_boundaries": {
            "online_reduced_solver_implementation_authorized": False,
            "truth_campaign_authorized": False,
            "physical_microburst_authorized": False,
            "exploratory_cycle_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "counts": {
            "branch_anchor_slots": len(branch["anchor_schedule"]),
            "transition_slots": len(transition["transition_schedule"]),
            "candidate_memory_orders": len(MEMORY_ORDERS),
        },
    }


def _checks(branch: dict, transition: dict, campaign: dict) -> dict[str, bool]:
    branch_slots = branch["anchor_schedule"]
    transition_slots = transition["transition_schedule"]
    return {
        "two_branches": tuple(branch["branches"]) == BRANCHES,
        "branch_slot_count": len(branch_slots) == 12,
        "branch_training_count": len(branch["training_ids"]) == 8,
        "branch_heldout_count": len(branch["sealed_heldout_ids"]) == 4,
        "branch_fine_count": sum(item["fine_layout"] for item in branch_slots) == 2,
        "memory_orders": tuple(branch["stable_memory"]["orders"]) == MEMORY_ORDERS,
        "memory_sealed": branch["stable_memory"]["freeze_before_opening_heldout"],
        "face_flux_form": branch["local_conservative_closure"][
            "fit_object"
        ]
        == "one_single_valued_face_flux_per_radial_face",
        "two_transition_directions": tuple(transition["directions"]) == TRANSITIONS,
        "transition_slot_count": len(transition_slots) == 8,
        "transition_training_count": sum(
            item["role"] == "training" for item in transition_slots
        )
        == 6,
        "transition_heldout_count": sum(
            item["role"] == "heldout" for item in transition_slots
        )
        == 2,
        "transition_online_microsteps_forbidden": transition[
            "online_fast_microsteps"
        ]
        == 0,
        "candidate_preflight_no_truth": campaign["next_package"][
            "new_exact_rate_calls"
        ]
        == 0,
        "candidate_preflight_no_roots": campaign["next_package"][
            "new_nonlinear_roots"
        ]
        == 0,
        "online_not_authorized": not campaign["authorization_boundaries"][
            "online_reduced_solver_implementation_authorized"
        ],
        "cycle_not_authorized": not campaign["authorization_boundaries"][
            "predictive_cycle_authorized"
        ],
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
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
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("hybrid branch-transition atlas manifest already exists")
    branch = _branch_contract()
    transition = _transition_contract()
    campaign = _campaign_contract(branch, transition)
    checks = _checks(branch, transition, campaign)
    if not all(checks.values()):
        raise RuntimeError(f"hybrid atlas definitions failed: {checks}")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(CANONICAL_DIRECTORY / "slow_branch_contract.json", branch)
    _write(CANONICAL_DIRECTORY / "fast_transition_contract.json", transition)
    _write(CANONICAL_DIRECTORY / "campaign_contract.json", campaign)
    _write(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "parent_classification": frozen["summary"]["classification"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "branch_and_transition_datasets_separated": True,
        "branch_anchor_slots": len(branch["anchor_schedule"]),
        "transition_slots": len(transition["transition_schedule"]),
        "candidate_memory_orders": list(MEMORY_ORDERS),
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "truth_campaign_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(
        CANONICAL_DIRECTORY / "checks.json",
        {"checks": checks, "passed": True},
    )
    source_files = (THIS_RUNNER, THIS_TEST, parent.THIS_RUNNER, parent.THIS_TEST)
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in parent.parent.field_manifest.training._thread_environment()
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Hybrid branch-transition atlas manifest WP10c9d6c7c3b5c4f25db",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The database is now split into two different mathematical objects. Cold/hot slow branches use stable quasi-steady graphs, conservative single-valued face-flux laws, and stable passive memory. Fast transitions use entry-to-exit impulse maps with integrated conservative ledgers.",
                "",
                "The validated forward patch remains an unclassified transition seed. It is not relabeled as a cold or hot branch and is not used to fit branch memory.",
                "",
                "The online solver will never integrate the 470-coordinate field or a fast transition ODE. It advances the branch model with multi-second macrosteps, brackets an event surface, and applies one prevalidated conservative reset map.",
                "",
                "The next package inventories existing accepted states and freezes branch/event candidates without new truth. Only after that no-truth preflight passes may a one-case branch-existence pilot be defined.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. No truth campaign, online solver, exploratory cycle, predictive cycle, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
