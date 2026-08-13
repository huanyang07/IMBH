#!/usr/bin/env python3
"""Freeze the face-36 augmented observable-memory screen."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_existing_state_overlap_consistency_preflight_wp10c9d6c7c3b5c4f11 as c4f11  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f12"
ARTIFACT = "causal_inner_face36_augmented_memory_screen_manifest_wp10c9d6c7c3b5c4f12"
THIS_RUNNER = "scripts/run_causal_inner_face36_augmented_memory_screen_manifest_wp10c9d6c7c3b5c4f12.py"
THIS_TEST = "tests/test_causal_inner_face36_augmented_memory_screen_manifest_wp10c9d6c7c3b5c4f12.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_FACE36_AUGMENTED_MEMORY_SCREEN_MANIFEST_WP10C9D6C7C3B5C4F12_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "memory_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest():
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": "face36_augmented_memory_screen_frozen_analysis_only_propagation_authorized",
        "definitions_only": True,
        "propagation_executed": False,
        "new_nonlinear_trajectory": False,
        "time_interval_seconds": [0.005, 0.020],
        "layouts": {
            "middle": "binding_primary",
            "fine": "cross_resolution_confirmation",
            "coarse": "absolute_scale_and_existing_response_reference_only",
        },
        "slow_coordinate": {
            "Q3_macro": ["exact_mapped_M_cells_36_to_64", "exact_mapped_J_cells_36_to_64", "exact_mapped_E_cells_36_to_64"],
            "responsive_height_history_is_not_an_absolute_Q_coordinate": True,
            "raw_face48_flux_forbidden": True,
        },
        "initial_lifts": {
            "structured_directions": 5,
            "seeded_smooth_random_directions": 24,
            "seed": 20260813,
            "initial_Q3_macro_null_projection_only": True,
            "projection_metric": "declared_primitive_column_scaling_with_exact_mapped_storage_constraint",
            "per_step_reprojection": False,
            "Euclidean_state_projection": False,
            "physical_fixed_Q_constraint_imposed": False,
        },
        "binding_output": {
            "name": "shared_face36_M_J_E_flux",
            "forms": ["instantaneous", "cumulative", "window_mean"],
            "analytic_JVP_from_shared_face_flux_map": True,
        },
        "augmented_diagnostic_output": {
            "guard_parent_mapped_storage": "cells_36_to_48_M_J_E",
            "guard_parent_responsive_height_history": "cells_36_to_48_angular_momentum_and_energy_history",
            "fine_complement_retained": True,
            "fine_complement_face36_observability_fraction": 0.060476803007652596,
        },
        "algorithm": {
            "reuse_committed_accepted_variable_step_BDF2_histories": True,
            "one_analytic_step_matrix_and_factorization_per_accepted_step": True,
            "all_29_directions_one_block_solve": True,
            "dense_full_propagator_forbidden": True,
            "output_weighted_time_stacked_SVD": True,
            "compute_finite_time_gain_regrowth_and_oscillation_diagnostics": True,
            "track_Q3_macro_leakage_without_suppressing_it": True,
        },
        "staging": {
            "run_middle_first": True,
            "stop_before_fine_on_any_method_gate_failure": True,
            "run_fine_only_after_middle_method_pass": True,
            "no_nonlinear_anchor_in_this_package": True,
        },
        "prospective_gates": {
            "maximum_initial_Q3_null_defect": 1.0e-10,
            "maximum_tangent_matrix_JVP_defect": 1.0e-8,
            "maximum_linear_solve_relative_defect": 1.0e-10,
            "maximum_component_closure_defect": 1.0e-12,
            "maximum_face36_output_map_defect": 1.0e-9,
            "minimum_cross_resolution_memory_subspace_cosine": 0.90,
            "minimum_observable_energy_capture": 0.99,
            "minimum_significant_singular_value_fraction": 1.0e-3,
            "rapid_contraction_final_to_peak_gain": 0.10,
            "rapid_contraction_last_quarter_regrowth_fraction": 0.05,
            "maximum_Q3_macro_leakage_for_unconstrained_architecture_classification": 0.10,
            "compact_retained_mode_limit": 3,
        },
        "decision": {
            "rapid_contraction_low_leakage": "authorize_definitions_only_quasi_steady_constraint_pilot_manifest",
            "one_to_three_persistent_modes_low_leakage": "authorize_definitions_only_retained_mode_Q_plus_a_pilot_manifest",
            "oscillatory_pair_low_leakage": "authorize_definitions_only_amplitude_phase_pilot_manifest",
            "broad_observable_memory_low_leakage": "authorize_definitions_only_HMM_microburst_manifest",
            "large_Q3_leakage": "derive_physical_constraint_reaction_map_before_conditional_memory_claim",
            "cross_resolution_failure": "memory_localization_only",
        },
        "cost_contract": {
            "expected_matrix_assembly_wall_hours_middle_plus_fine": [2.5, 4.5],
            "block_solve_marginal_cost_is_negligible": True,
            "no_50ms_or_full_fine_nonlinear_run": True,
            "checkpoint_each_accepted_step": True,
        },
        "interpretation_limits": {
            "this_is_not_a_fixed_Q_attractor_experiment": True,
            "this_cannot_prove_a_quasi_steady_manifold": True,
            "persistent_modes_select_architecture_not_fitted_coefficients": True,
            "absolute_face48_export_rejection_preserved": True,
        },
        "hard_stops": [
            "do_not_project_each_step_without_a_physical_reaction_map",
            "do_not_use_face48_as_slow_output",
            "do_not_discard_the_guard_fine_complement",
            "do_not_fit_a_scalar_relaxation_law_before_the_spectrum_is_known",
            "do_not_run_50ms_fixed_Q_or_reduced_evolution",
        ],
        "memory_propagation_authorized": True,
        "new_nonlinear_trajectory_authorized": False,
        "fixed_Q_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f13_face36_augmented_analysis_only_memory_screen",
    }


def _catalog(summary):
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "PROSPECTIVE"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "latest_work_package": WORK_PACKAGE})
    _write(CANONICAL_SUMMARY, catalog)


def main():
    parent = _read(c4f11.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["method_gates_passed"]
        or not parent["spatial_gates_passed"]
        or not parent["guard_reaction_observability_gate_passed"]
        or parent["authorized_next"] != "WP10c9d6c7c3b5c4f12_definitions_only_face36_augmented_projected_memory_screen_manifest"
    ):
        raise RuntimeError("c4f12 authorization changed")
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "propagation_executed": False,
        "face36_absolute_and_overlap_gates_preserved": True,
        "face48_absolute_export_rejection_preserved": True,
        "memory_propagation_authorized": True,
        "new_nonlinear_trajectory_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(CONFIG_PATH, {"schema_version": SCHEMA_VERSION, "time_interval_seconds": manifest["time_interval_seconds"], "shared_face": 36, "directions": 29, "layouts": ["middle", "fine"]})
    _write(MANIFEST_PATH, manifest)
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Face-36 augmented memory-screen manifest\n\n"
        f"Classification: `{summary['classification']}`.\n\n"
        "This definitions-only package authorizes a block-tangent analysis on the committed middle and fine BDF2 histories. It uses exact macro Q3-null initial lifts, the convergent face-36 M/J/E flux, and retained guard storage/history diagnostics. It imposes no per-step projection or fixed-Q reaction.\n\n"
        "The run is expected to cost a few hours of matrix assembly, not a new nonlinear or 50 ms campaign.\n",
        encoding="utf-8",
    )
    _write(PROVENANCE_PATH, {"schema_version": SCHEMA_VERSION, "parent_summary_sha256": _sha(c4f11.SUMMARY_PATH), "source_hashes": {THIS_RUNNER: _sha(ROOT / THIS_RUNNER), THIS_TEST: _sha(ROOT / THIS_TEST) if (ROOT / THIS_TEST).exists() else None}})
    files = (CONFIG_PATH, MANIFEST_PATH, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(path)}  {path.name}\n" for path in files), encoding="utf-8")
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
