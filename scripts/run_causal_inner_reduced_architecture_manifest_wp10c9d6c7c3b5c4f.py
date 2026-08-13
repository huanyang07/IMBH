#!/usr/bin/env python3
"""Freeze the reduced-architecture and observable-memory screen contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_final_three_grid_20ms_spatial_reanalysis_wp10c9d6c7c3b5c4e12 as c4e12  # noqa: E402
import run_causal_inner_nonlinear_optimized_middle_20ms_completion_wp10c9d6c7c3b5c4e3 as c4e3  # noqa: E402
import run_causal_inner_nonlinear_middle_6ms_spatial_cost_pilot_wp10c9d6c7c3b5c4e1 as c4e1  # noqa: E402
import run_causal_inner_nonlinear_cost_bounded_fine_20ms_base_tangent_wp10c9d6c7c3b5c4e8 as c4e8  # noqa: E402
import run_causal_inner_nonlinear_twenty_ms_completion_wp10c9d6c7c3b5c4c1 as c4c1  # noqa: E402
import run_causal_inner_nonlinear_ten_ms_screen_wp10c9d6c7c3b5c4b2 as c4b2  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f"
ANALYZED_CERTIFICATE_COMMIT = "3bb81e72d0673a391a8a8f6d8e33bcc9827a0d91"
ANALYZED_CERTIFICATE_PARENT = "1ed136d6d813e94e7edc910f1a7e6c778aed8e45"
ANALYZED_CERTIFICATE_TREE = "a589faf8a9b5cfedb1172d57d443af338d2b7cb3"

ARTIFACT = "causal_inner_reduced_architecture_manifest_wp10c9d6c7c3b5c4f"
THIS_RUNNER = (
    "scripts/run_causal_inner_reduced_architecture_manifest_"
    "wp10c9d6c7c3b5c4f.py"
)
THIS_TEST = (
    "tests/test_causal_inner_reduced_architecture_manifest_"
    "wp10c9d6c7c3b5c4f.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_REDUCED_ARCHITECTURE_MANIFEST_"
    "WP10C9D6C7C3B5C4F_2026-08-13.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "architecture_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

COARSE_ARRAYS = c4c1.DECISIVE_ARRAYS
COARSE_EARLY_ARRAYS = c4b2.DECISIVE_ARRAYS
MIDDLE_ARRAYS = c4e3.DECISIVE_ARRAYS
MIDDLE_PILOT_ARRAYS = c4e1.DECISIVE_ARRAYS
FINE_ARRAYS = c4e8.DECISIVE_ARRAYS
FINAL_ARRAYS = c4e12.DECISIVE_ARRAYS
EXTRACTION_RADIUS_RG = 1.9531594414758637
EXTRACTION_FACE_INDICES = (2, 4, 8)
COUPLING_FACE_INDICES = (48, 96, 192)
LAYOUTS = ("coarse", "middle", "fine")
CONSERVATIVE_MAPPED_ROWS = (0, 2, 3)
RANDOM_SEED = 20260813
RANDOM_LIFT_COUNT = 24
RANDOMIZED_OVERSAMPLING = 8


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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _npz_keys(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as payload:
        return set(payload.files)


def _validate_parent() -> dict:
    summary = _read_json(c4e12.SUMMARY_PATH)
    if (
        not summary["passed"]
        or not summary["fine_twenty_ms_spatial_certificate_issued"]
        or not summary["state_twenty_ms_spatial_contract_certified"]
        or not summary["extraction_twenty_ms_spatial_contract_certified"]
        or not summary["reduced_architecture_manifest_authorized"]
        or summary["fixed_q_micro_solver_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or summary["physical_failure_detected"]
    ):
        raise RuntimeError("c4f predecessor authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_CERTIFICATE_COMMIT)
        != ANALYZED_CERTIFICATE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_CERTIFICATE_COMMIT}^")
        != ANALYZED_CERTIFICATE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_CERTIFICATE_COMMIT}^{{tree}}")
        != ANALYZED_CERTIFICATE_TREE
    ):
        raise RuntimeError("c4f analyzed certificate identity changed")
    required = {
        MIDDLE_ARRAYS: {
            "base__accepted_times",
            "base__accepted_states",
            "base__accepted_primitive_histories",
            "base__accepted_mapped_histories",
            "base__accepted_height_histories",
            "base__accepted_previous_timesteps",
            "extraction__base_values",
        },
        MIDDLE_PILOT_ARRAYS: {
            "base__accepted_times",
            "base__accepted_states",
            "base__accepted_primitive_histories",
            "base__accepted_mapped_histories",
            "base__accepted_height_histories",
            "base__accepted_previous_timesteps",
        },
        FINE_ARRAYS: {
            "base__accepted_times",
            "base__accepted_states",
            "base__accepted_primitive_histories",
            "base__accepted_mapped_histories",
            "base__accepted_height_histories",
            "base__accepted_previous_timesteps",
            "extraction__base_values",
        },
        COARSE_ARRAYS: {
            "base_main__output_times",
            "base_main__output_states",
            "base_main__output_extraction_partition",
        },
        COARSE_EARLY_ARRAYS: {
            "base_main__output_times",
            "base_main__output_states",
            "base_main__output_extraction_partition",
        },
        FINAL_ARRAYS: {
            "coarse_state_response",
            "middle_state_response",
            "fine_nonlinear_state_response",
            "coarse_extraction_response",
            "middle_extraction_response",
            "fine_nonlinear_extraction_response",
        },
    }
    for path, names in required.items():
        missing = names - _npz_keys(path)
        if missing:
            raise RuntimeError(f"c4f canonical input is incomplete: {path}: {missing}")
    return summary


def _manifest() -> dict:
    observable_names = tuple(c4e12.OBSERVABLE_NAMES)
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "reduced_architecture_manifest_frozen_analysis_only_observable_"
            "memory_screen_authorized"
        ),
        "definitions_only": True,
        "propagation_executed": False,
        "physical_operator_changed": False,
        "production_defaults_changed": False,
        "certified_parent_scope": {
            "nonlinear_truth_model_spatially_certified_through_seconds": 0.020,
            "state_and_extraction_response_certificate": True,
            "physical_failure_detected": False,
            "slow_export": "certified_conservative_exterior_partition",
            "extraction_radius_rg": EXTRACTION_RADIUS_RG,
            "layout_extraction_faces": dict(
                zip(LAYOUTS, EXTRACTION_FACE_INDICES, strict=True)
            ),
            "layout_coupling_faces": dict(
                zip(LAYOUTS, COUPLING_FACE_INDICES, strict=True)
            ),
            "raw_pointwise_horizon_face_flux_rejected": True,
            "excision_to_extraction_buffer_remains_in_microdomain": True,
        },
        "absolute_baseline_audit": {
            "required_before_closure_fit": True,
            "uses_existing_canonical_trajectories_only": True,
            "layouts": LAYOUTS,
            "channels": (
                "absolute_primitive_state_on_common_parent",
                "absolute_mapped_exterior_M_J_E_storage",
                "absolute_instantaneous_extraction_partition",
                "absolute_cumulative_extraction_partition",
                "absolute_window_mean_extraction_partition",
                "baseline_plus_response_reconstruction",
            ),
            "responsive_height_temporal_one_form_is_not_an_absolute_state": True,
            "minimum_spatial_RMS_order": 0.75,
            "minimum_spatial_error_direction_cosine": 0.90,
            "relative_observability_floor": 1.0e-10,
            "below_floor_route": "report_upper_bound_without_order_claim",
            "baseline_plus_response_relative_defect_gate": 1.0e-12,
            "no_new_trajectory": True,
        },
        "slow_coordinate_candidates": {
            "Q3": {
                "names": (
                    "exterior_mapped_mass",
                    "exterior_mapped_angular_momentum",
                    "exterior_mapped_killing_energy",
                ),
                "mapped_storage_rows": CONSERVATIVE_MAPPED_ROWS,
                "definition": (
                    "sum_the_exact_cell_integrated_mapped_conserved_state_"
                    "over_cells_between_extraction_and_coupling_faces"
                ),
                "single_valued_state_function": True,
                "responsive_height_one_form_excluded": True,
            },
            "Q4": {
                "extends": "Q3",
                "additional_name": "exterior_column_thermal_content",
                "definition": (
                    "same_cell_quadrature_sum_of_surface_density_times_"
                    "specific_internal_energy_over_the_exterior_partition"
                ),
                "single_valued_state_function": True,
                "selection_status": "candidate_not_promoted",
            },
            "Q5": {
                "extends": "Q4",
                "additional_name": "exterior_relaxing_stress_storage",
                "definition": (
                    "same_cell_quadrature_sum_of_the_fifth_exact_mapped_"
                    "conserved_component_over_the_exterior_partition"
                ),
                "mapped_storage_row": 4,
                "single_valued_state_function": True,
                "selection_status": "candidate_not_promoted",
            },
            "selection_rule": (
                "choose_the_smallest_nested_Q_whose_held_out_equal_Q_output_"
                "memory_passes_the_prospective_observability_contract"
            ),
            "staged_screen_order": ("Q3", "Q4", "Q5"),
            "first_analysis_package_tests_Q3_only": True,
            "later_candidate_runs_require_a_fresh_authorization_from_the_"
            "preceding_result": True,
            "post_result_coordinate_addition_forbidden": True,
        },
        "augmented_discrete_state": {
            "required": True,
            "components": (
                "current_primitive_state",
                "previous_primitive_increment",
                "previous_mapped_storage_increment",
                "previous_responsive_height_storage_increment",
                "previous_timestep_seconds",
            ),
            "primitive_snapshot_only_analysis_forbidden": True,
            "responsive_height_history_retained_because_one_form_is_nonexact": True,
            "variable_step_BDF2_history_ratios_retained": True,
        },
        "equal_Q_screen": {
            "classification": "kinematic_observable_memory_screen_not_fixed_Q_dynamics",
            "initial_directions_satisfy": "DQ_at_initial_state_times_delta_p_equals_zero",
            "constraint_null_basis": "physically_scaled_weighted_SVD",
            "propagation": (
                "unconstrained_augmented_discrete_BDF_tangent_along_the_"
                "committed_5_to_20ms_base_trajectory"
            ),
            "measure_slow_leakage_without_reprojecting_each_step": True,
            "per_step_reprojection_forbidden": True,
            "may_not_be_called_a_fixed_Q_attractor_test": True,
        },
        "future_fixed_Q_constraint": {
            "not_authorized_in_this_package": True,
            "required_form": "block_system_[M,-B_Q;DQ,0]_[p_dot,lambda]=[-R,0]",
            "B_Q_must_represent_physical_reservoir_or_constraint_reaction": True,
            "B_Q_equals_DQ_transpose_may_not_be_assumed": True,
            "metric_units_and_rank_must_be_certified": True,
            "all_multiplier_M_J_E_and_height_work_must_be_ledgered": True,
            "manual_primitive_freezing_forbidden": True,
        },
        "lift_ensemble": {
            "deterministic_seed": RANDOM_SEED,
            "random_equal_Q_directions": RANDOM_LIFT_COUNT,
            "randomized_SVD_oversampling": RANDOMIZED_OVERSAMPLING,
            "structured_directions": (
                "inward_acoustic",
                "outward_acoustic",
                "material_contact",
                "inward_shear",
                "outward_shear",
                "generic_five_field",
                "leading_transient_growth",
                "fine_grid_complement",
            ),
            "training_and_held_out_direction_hashes_required": True,
            "all_directions_scaled_in_declared_physical_metric": True,
            "full_nonlinear_lift_propagation": False,
        },
        "observable_memory_analysis": {
            "time_interval_seconds": (0.005, 0.020),
            "binding_primary_layout": "middle",
            "coarse_and_fine_roles": "cross_resolution_and_fine_complement_checks",
            "outputs": observable_names,
            "historical_output_label_alias": (
                "inner_flux_in_the_committed_13_vector_means_the_fixed_"
                "extraction_surface_flux_not_the_raw_excision_face_flux"
            ),
            "forms": ("instantaneous", "cumulative", "window_mean"),
            "algorithms": (
                "matrix_free_block_augmented_BDF_propagation",
                "randomized_output_weighted_SVD",
                "finite_time_singular_values_and_transient_growth",
                "principal_angles_across_time_and_resolution",
                "decay_regrowth_and_oscillation_diagnostics",
            ),
            "dense_full_propagator_or_dense_Gramian_forbidden": True,
            "one_step_matrix_factorization_reused_for_all_directions": True,
            "existing_tangent_matrix_JVP_gate": 1.0e-8,
            "linear_solve_relative_defect_gate": 1.0e-10,
            "constraint_null_defect_gate": 1.0e-10,
            "minimum_observable_energy_capture": 0.99,
            "compact_retained_mode_limit": 3,
            "minimum_cross_resolution_subspace_cosine": 0.90,
            "maximum_temporal_fraction_of_memory_signal": 0.10,
            "rapid_contraction_final_to_peak_gain": 0.10,
            "rapid_contraction_last_quarter_regrowth_fraction": 0.05,
            "persistent_mode_final_to_peak_floor": 0.10,
            "minimum_significant_singular_value_fraction": 1.0e-3,
            "Q3_failure_authorizes_only_Q4_definitions_or_screen_manifest": True,
        },
        "fine_complement_contract": {
            "definition": "fine_state_minus_prolongated_restricted_fine_state",
            "state_energy_and_extraction_observability_both_required": True,
            "maximum_discarded_output_fraction": 0.01,
            "maximum_fraction_of_middle_fine_spatial_difference": 0.10,
            "failed_gate_requires_retention_or_targeted_fine_followup": True,
        },
        "decision_tree": {
            "rapid_unique_observable_contraction": (
                "authorize_definitions_only_quasi_steady_fixed_Q_pilot_manifest"
            ),
            "one_to_three_persistent_observable_modes": (
                "authorize_definitions_only_retained_mode_Q_plus_a_pilot_manifest"
            ),
            "one_cross_resolution_stable_oscillatory_pair": (
                "authorize_definitions_only_amplitude_phase_pilot_manifest"
            ),
            "broad_but_decaying_or_mixing_observable_memory": (
                "authorize_definitions_only_HMM_microburst_pilot_manifest"
            ),
            "multiple_persistent_branches": (
                "authorize_definitions_only_hysteretic_branch_pilot_manifest"
            ),
            "no_compact_cross_resolution_observable_structure": (
                "retain_inner_micro_solver_and_reject_compact_autonomous_ROM"
            ),
            "screen_inconclusive_at_20ms": (
                "authorize_only_a_targeted_duration_or_constrained_pilot_manifest"
            ),
        },
        "cost_contract": {
            "new_nonlinear_trajectories": 0,
            "new_physical_propagation": False,
            "reuse_middle_and_fine_accepted_states_and_BDF_histories": True,
            "matrix_assemblies_once_per_selected_base_step": True,
            "block_right_hand_sides": True,
            "estimated_wall_hours": (2.0, 8.0),
            "cost_estimate_is_scheduling_only": True,
            "stop_before_dense_operator_or_unplanned_long_duration": True,
        },
        "hard_stops": (
            "do_not_run_50ms_or_125ms_trajectory",
            "do_not_start_constrained_fixed_Q_propagation",
            "do_not_start_reduced_slow_evolution",
            "do_not_fit_a_scalar_relaxation_law_before_memory_classification",
            "do_not_treat_kinematic_projection_as_fixed_Q_dynamics",
            "do_not_drop_BDF_or_responsive_height_history",
            "do_not_use_raw_pointwise_horizon_face_flux_as_slow_export",
            "do_not_change_the_certified_operator_or_spatial_gates",
            "do_not_add_slow_coordinates_after_inspecting_held_out_results",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c4f1_analysis_only_absolute_baseline_and_"
            "observable_memory_screen"
        ),
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
                    "sha256": _sha256(path),
                    "scientific_status": "PROSPECTIVE",
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
    catalog = _read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": ANALYZED_CERTIFICATE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def main() -> int:
    parent = _validate_parent()
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent["classification"],
        "twenty_ms_spatial_certificate_preserved": True,
        "raw_pointwise_horizon_face_rejection_preserved": True,
        "absolute_baseline_audit_frozen": True,
        "nested_Q3_Q4_Q5_candidates_frozen": True,
        "augmented_BDF_memory_contract_frozen": True,
        "kinematic_screen_explicitly_not_fixed_Q_dynamics": True,
        "analysis_only_memory_screen_authorized": True,
        "new_trajectory_authorized": False,
        "fifty_ms_manifest_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "layouts": LAYOUTS,
            "time_interval_seconds": (0.005, 0.020),
            "extraction_radius_rg": EXTRACTION_RADIUS_RG,
            "extraction_face_indices": EXTRACTION_FACE_INDICES,
            "coupling_face_indices": COUPLING_FACE_INDICES,
            "random_seed": RANDOM_SEED,
            "random_equal_Q_directions": RANDOM_LIFT_COUNT,
            "randomized_SVD_oversampling": RANDOMIZED_OVERSAMPLING,
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE",
            "analyzed_certificate_commit": ANALYZED_CERTIFICATE_COMMIT,
            "analyzed_certificate_parent_commit": ANALYZED_CERTIFICATE_PARENT,
            "analyzed_certificate_tree_sha": ANALYZED_CERTIFICATE_TREE,
            "analysis_execution_head": _git_value("rev-parse", "HEAD"),
            "packaging_commit": None,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
            "input_hashes": {
                "final_summary": _sha256(c4e12.SUMMARY_PATH),
                "final_arrays": _sha256(FINAL_ARRAYS),
                "coarse_arrays": _sha256(COARSE_ARRAYS),
                "coarse_early_arrays": _sha256(COARSE_EARLY_ARRAYS),
                "middle_arrays": _sha256(MIDDLE_ARRAYS),
                "middle_pilot_arrays": _sha256(MIDDLE_PILOT_ARRAYS),
                "fine_arrays": _sha256(FINE_ARRAYS),
            },
            "implementation_source_hashes": {
                path: _sha256(ROOT / path)
                for path in (THIS_RUNNER, THIS_TEST)
                if (ROOT / path).exists()
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Reduced-architecture manifest WP10c9d6c7c3b5c4f",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                "The three-grid nonlinear 20 ms state and conservative exterior-partition certificate is preserved. No physical failure is detected, and the rejected raw pointwise horizon-face flux is not used as a slow observable.",
                "",
                "This package freezes nested exact-state candidates `Q3/Q4/Q5`, the complete augmented variable-step BDF2 memory state, a kinematic equal-Q observability screen, fine-complement tests, prospective memory-dimension decisions, and cost stops. The responsive-height temporal contribution is a non-exact one-form and is therefore retained as BDF history rather than misidentified as an absolute slow coordinate.",
                "",
                "The next package is analysis-only and must reuse the committed coarse/middle/fine trajectories. It may measure observable memory and slow leakage, but it is not a constrained fixed-Q attractor experiment.",
                "",
                "No 50 ms trajectory, constrained fixed-Q propagation, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("architecture_manifest.json", "config.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
