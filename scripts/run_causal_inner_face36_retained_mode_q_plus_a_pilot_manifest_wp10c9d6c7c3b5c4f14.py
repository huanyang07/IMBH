#!/usr/bin/env python3
"""Freeze the face-36 retained-mode Q+a pilot contract.

Definitions only: no tangent, nonlinear, fixed-Q, 50 ms, or reduced slow
trajectory is advanced by this package.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13 as c4f13  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f14"
ARTIFACT = "causal_inner_face36_retained_mode_q_plus_a_pilot_manifest_wp10c9d6c7c3b5c4f14"
THIS_RUNNER = "scripts/run_causal_inner_face36_retained_mode_q_plus_a_pilot_manifest_wp10c9d6c7c3b5c4f14.py"
THIS_TEST = "tests/test_causal_inner_face36_retained_mode_q_plus_a_pilot_manifest_wp10c9d6c7c3b5c4f14.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_FACE36_RETAINED_MODE_Q_PLUS_A_PILOT_MANIFEST_WP10C9D6C7C3B5C4F14_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "pilot_manifest.json"
MODE_BASIS_PATH = CANONICAL_DIRECTORY / "mode_basis.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

FORMS = ("instantaneous", "cumulative", "window_mean")
LAYOUTS = ("middle", "fine")
MODE_DIMENSION = 2


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent() -> dict:
    summary = _read(c4f13.SUMMARY_PATH)
    if (
        not summary["passed"]
        or summary["classification"]
        != "face36_compact_persistent_observable_memory_detected"
        or summary["authorized_next"]
        != "definitions_only_retained_mode_Q_plus_a_pilot_manifest"
        or summary["fixed_Q_micro_solver_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or summary["fifty_ms_propagation_authorized"]
        or summary["physical_failure_detected"]
    ):
        raise RuntimeError("c4f14 authorization changed")
    for label in LAYOUTS:
        memory = summary[label]["memory"]
        if (
            memory["maximum_binding_k99"] != MODE_DIMENSION
            or memory["rapid_contraction"]
            or memory["maximum_Q3_macro_leakage"] > 0.10
        ):
            raise RuntimeError(f"c4f14 {label} memory classification changed")
    if summary["cross_resolution"]["minimum_principal_cosine"] < 0.90:
        raise RuntimeError("c4f14 memory subspace is not cross-grid stable")
    return summary


def _normalized_gram(singular: np.ndarray, right: np.ndarray) -> np.ndarray:
    gram = (right.T * np.square(singular)) @ right
    trace = float(np.trace(gram))
    if trace <= np.finfo(float).tiny:
        raise RuntimeError("c4f14 observable Gramian is singular")
    return gram / trace


def _mode_basis() -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(c4f13.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        layout_grams = {}
        form_grams = {}
        for label in LAYOUTS:
            layout_gram = np.zeros((c4f13.TOTAL_DIRECTIONS,) * 2)
            for form in FORMS:
                gram = _normalized_gram(
                    arrays[f"{label}__{form}_singular_values"],
                    arrays[f"{label}__{form}_right_vectors"],
                )
                form_grams[(label, form)] = gram
                layout_gram += gram / len(FORMS)
            layout_grams[label] = layout_gram
        consensus_gram = sum(layout_grams.values()) / len(LAYOUTS)
        values, vectors = np.linalg.eigh(consensus_gram)
        order = np.argsort(values)[::-1]
        values = values[order]
        basis = vectors[:, order[:MODE_DIMENSION]]
        for mode in range(MODE_DIMENSION):
            pivot = int(np.argmax(np.abs(basis[:, mode])))
            if basis[pivot, mode] < 0.0:
                basis[:, mode] *= -1.0
        captures = {}
        layout_cosines = {}
        for label in LAYOUTS:
            local_values, local_vectors = np.linalg.eigh(layout_grams[label])
            local = local_vectors[:, np.argsort(local_values)[::-1][:MODE_DIMENSION]]
            layout_cosines[label] = np.linalg.svd(
                basis.T @ local, compute_uv=False
            )
            for form in FORMS:
                gram = form_grams[(label, form)]
                captures[(label, form)] = float(
                    np.trace(basis.T @ gram @ basis) / np.trace(gram)
                )
    metrics = {
        "consensus_energy_capture": float(
            np.sum(values[:MODE_DIMENSION]) / np.sum(values)
        ),
        "spectral_gap_sigma2_over_sigma3": float(values[1] / values[2]),
        "minimum_form_energy_capture": min(captures.values()),
        "minimum_layout_subspace_cosine": min(
            float(np.min(value)) for value in layout_cosines.values()
        ),
        "form_energy_capture": {
            label: {form: captures[(label, form)] for form in FORMS}
            for label in LAYOUTS
        },
        "layout_subspace_cosines": {
            label: value.tolist() for label, value in layout_cosines.items()
        },
    }
    stored = {
        "consensus_direction_coefficients": basis,
        "consensus_gram_eigenvalues": values,
        "middle_normalized_gram": layout_grams["middle"],
        "fine_normalized_gram": layout_grams["fine"],
    }
    return metrics, stored


def _manifest(parent: dict, basis_metrics: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "face36_retained_mode_Q_plus_a_pilot_manifest_frozen_"
            "reaction_map_preflight_authorized"
        ),
        "definitions_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "certified_parent_result": {
            "classification": parent["classification"],
            "observable_memory_dimension": MODE_DIMENSION,
            "minimum_middle_fine_memory_subspace_cosine": parent[
                "cross_resolution"
            ]["minimum_principal_cosine"],
            "rapid_contraction_detected": False,
            "oscillatory_pair_resolution_confirmed": False,
            "raw_face48_export_rejection_preserved": True,
        },
        "physical_partition": {
            "inner_micro_core_parent_cells": [0, 36],
            "macro_exterior_parent_cells": [36, 64],
            "duplicate_micro_guard_parent_cells": [36, 48],
            "macro_only_reaction_support_parent_cells": [48, 64],
            "shared_exchange_parent_face": 36,
            "binding_exchange": "shared_face36_M_J_E_flux",
            "macro_owns_guard_physical_inventory_exactly_once": True,
            "micro_guard_fine_complement_retained": True,
        },
        "slow_state": {
            "Q": [
                "exact_mapped_M_parent_cells_36_to_64",
                "exact_mapped_J_parent_cells_36_to_64",
                "exact_mapped_E_parent_cells_36_to_64",
            ],
            "responsive_height_history_is_not_an_absolute_Q_coordinate": True,
            "evolution_exchange": "shared_face36_M_J_E_flux_plus_macro_sources",
            "raw_face48_flux_forbidden": True,
        },
        "retained_amplitudes": {
            "names": ["a1", "a2"],
            "dimension": MODE_DIMENSION,
            "basis": (
                "equal_weight_consensus_of_middle_and_fine_normalized_"
                "instantaneous_cumulative_window_mean_face36_Gramians"
            ),
            "consensus_basis_metrics": basis_metrics,
            "state_lifts": (
                "reconstruct_from_the_frozen_29_initial_Q3_null_lifts_and_"
                "consensus_direction_coefficients"
            ),
            "dual_coordinate": (
                "descriptor_weighted_Petrov_Galerkin_dual_A_satisfying_"
                "A_times_state_lifts_equals_I_and_A_times_DQ_dual_equals_0"
            ),
            "normalization": "dimensionless_with_a1_a2_equal_to_consensus_coefficients_at_5ms",
            "sign_rule": "largest_absolute_consensus_coefficient_is_positive",
            "not_assumed": [
                "eigenmodes",
                "normal_modes",
                "oscillatory_phase_pair",
                "individually_conserved_quantities",
            ],
            "time_dependent_basis_forbidden_in_first_pilot": True,
        },
        "physical_constraint_reaction": {
            "constraint": "DQ_macro_times_p_dot_equals_zero_during_fixed_Q_microburst",
            "reaction_map_name": "B_Q",
            "reaction_support": "macro_only_parent_cells_48_to_64",
            "reaction_envelope": (
                "nonnegative_sin_squared_cell_volume_weight_over_parent_"
                "cells_48_to_64_normalized_to_unit_integral"
            ),
            "raw_physical_reaction_channels": {
                "mass_loading": (
                    "unit_mapped_mass_with_background_local_specific_"
                    "angular_momentum_and_Killing_energy"
                ),
                "external_torque": (
                    "unit_mapped_angular_momentum_with_local_Omega_times_"
                    "torque_Killing_energy_work"
                ),
                "external_heating": "unit_mapped_Killing_energy_only",
            },
            "row_units": "exact_monolithic_mapped_M_J_E_storage_rate_conventions",
            "normalized_map": (
                "B_Q_equals_B_raw_times_inverse_of_DQ_M_inverse_B_raw"
            ),
            "normalization": "DQ_macro_times_M_inverse_times_B_Q_equals_I3",
            "KKT_system": [
                "M_times_p_dot_minus_B_Q_times_lambda_equals_minus_R",
                "DQ_macro_times_p_dot_equals_zero",
            ],
            "B_Q_equals_DQ_transpose_forbidden": True,
            "Euclidean_projection_forbidden": True,
            "reaction_must_not_modify_micro_core_or_duplicate_guard": True,
            "reaction_M_J_E_and_work_must_be_ledgered": True,
            "responsive_height_BDF_history_must_be_updated": True,
        },
        "reduced_candidate": {
            "state": "Q_plus_a1_plus_a2",
            "form": ["Q_dot=F_Q(Q,a)", "a_dot=F_a(Q,a)"],
            "linear_terms": "projected_discrete_BDF_tangent_preflight_only",
            "nonlinear_terms": "not_fitted_until_two_selected_nonlinear_anchors_pass",
            "face36_output_map": "Y=F_face36(Q,a)",
            "closure_residual": "full_micro_output_minus_Q_plus_a_reconstruction",
        },
        "authorized_reaction_map_preflight": {
            "uses_existing_middle_and_fine_5_to_20ms_states_only": True,
            "new_trajectory": False,
            "construct_DQ_macro_and_B_Q_at_times_ms": [5.0, 10.0, 16.0, 20.0],
            "verify_KKT_Schur_conditioning": True,
            "verify_exact_reaction_ledgers": True,
            "construct_state_lifts_and_descriptor_duals": True,
            "verify_two_mode_face36_reconstruction_for_all_29_directions": True,
            "verify_middle_fine_basis_and_coefficient_consistency": True,
            "verify_retained_guard_complement_is_not_discarded": True,
        },
        "prospective_preflight_gates": {
            "minimum_consensus_form_energy_capture": 0.99,
            "minimum_consensus_layout_subspace_cosine": 0.95,
            "minimum_mode_spectral_gap_sigma2_over_sigma3": 5.0,
            "maximum_DQ_M_inverse_BQ_identity_defect": 1.0e-10,
            "maximum_KKT_linear_solve_relative_defect": 1.0e-10,
            "maximum_reaction_ledger_relative_defect": 1.0e-12,
            "maximum_state_lift_Q3_defect": 1.0e-10,
            "maximum_dual_biorthogonality_defect": 1.0e-10,
            "maximum_two_mode_face36_output_weighted_RMS_error": 0.10,
            "maximum_two_mode_face36_significant_direction_error": 0.25,
            "significant_direction_relative_response_floor": 1.0e-3,
            "minimum_middle_fine_amplitude_history_cosine": 0.95,
            "maximum_KKT_Schur_condition_number": 1.0e8,
        },
        "conditional_nonlinear_pilot_after_preflight": {
            "layout": "middle",
            "Q_points": 1,
            "equal_Q_lifts_screened_by_block_tangent": 29,
            "maximum_full_nonlinear_anchor_lifts": 2,
            "anchor_selection": (
                "largest_predicted_two_mode_face36_effect_and_largest_"
                "predicted_closure_residual"
            ),
            "amplitudes": [0.5, 1.0],
            "signs": [-1, 1],
            "duration_selected_only_after_projected_decay_estimate": True,
            "fine_strategy": (
                "fine_KKT_tangent_and_short_shadow_first_full_fine_only_if_"
                "uncertainty_exceeds_10_percent_of_spatial_difference"
            ),
        },
        "future_nonlinear_gates": {
            "maximum_scaled_nonlinear_residual": 1.0e-10,
            "maximum_conservative_and_reaction_ledger_defect": 1.0e-12,
            "maximum_relative_Q_drift": 1.0e-8,
            "maximum_tangent_vs_nonlinear_state_response_fraction": 0.01,
            "maximum_tangent_vs_nonlinear_face36_response_fraction": 0.01,
            "maximum_two_mode_output_closure_error": 0.05,
            "minimum_middle_fine_state_and_face36_order": 0.75,
            "minimum_middle_fine_error_direction_cosine": 0.90,
            "maximum_temporal_or_surrogate_fraction_of_spatial_difference": 0.10,
            "restart_replay": "bitwise",
            "incoming_excision_characteristics": 0,
        },
        "decision": {
            "reaction_and_coordinate_preflight_passes": (
                "authorize_definitions_only_one_Q_constrained_nonlinear_"
                "retained_mode_pilot_manifest"
            ),
            "reaction_map_fails": "redesign_physical_macro_reservoir_coupling",
            "two_mode_reconstruction_fails": (
                "increase_output_oriented_memory_dimension_without_"
                "discarding_guard_state"
            ),
            "basis_cross_resolution_fails": "return_to_memory_basis_localization",
            "nonlinear_anchor_fails_later": "retain_inner_micro_solver_or_HMM",
        },
        "cost_contract": {
            "reaction_map_preflight_expected_wall_hours": [0.25, 1.0],
            "one_Q_middle_pilot_expected_wall_hours_after_authorization": [4.0, 12.0],
            "block_screen_before_nonlinear_anchors": True,
            "no_50ms_or_full_fine_trajectory": True,
            "no_large_rectangular_Q_database": True,
        },
        "hard_stops": [
            "do_not_run_fixed_Q_or_nonlinear_microburst_in_this_package",
            "do_not_assume_B_Q_equals_DQ_transpose",
            "do_not_call_the_two_modes_an_oscillatory_pair",
            "do_not_discard_or_double_count_the_guard_state",
            "do_not_use_face48_as_the_slow_exchange",
            "do_not_fit_nonlinear_reduced_coefficients_before_anchor_gates",
            "do_not_start_50ms_or_reduced_slow_evolution",
        ],
        "reaction_map_preflight_authorized": True,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f15_analysis_only_Q_plus_a_reaction_map_"
            "and_coordinate_preflight"
        ),
    }


def _catalog(summary: dict) -> None:
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
                    "scientific_status": "PROSPECTIVE",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
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
    parent = _validate_parent()
    basis_metrics, stored_basis = _mode_basis()
    manifest = _manifest(parent, basis_metrics)
    gates = manifest["prospective_preflight_gates"]
    if (
        basis_metrics["minimum_form_energy_capture"]
        < gates["minimum_consensus_form_energy_capture"]
        or basis_metrics["minimum_layout_subspace_cosine"]
        < gates["minimum_consensus_layout_subspace_cosine"]
        or basis_metrics["spectral_gap_sigma2_over_sigma3"]
        < gates["minimum_mode_spectral_gap_sigma2_over_sigma3"]
    ):
        raise RuntimeError("c4f14 consensus two-mode preflight failed")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "observable_memory_dimension": MODE_DIMENSION,
        "consensus_basis_metrics": basis_metrics,
        "physical_reaction_map_derived": False,
        "reaction_map_preflight_authorized": True,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "face48_absolute_export_rejection_preserved": True,
        "guard_complement_retained": True,
        "physical_failure_detected": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODE_BASIS_PATH, **stored_basis)
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "mode_dimension": MODE_DIMENSION,
            "parent_shared_face": 36,
            "parent_guard_cells": [36, 48],
            "parent_macro_cells": [36, 64],
            "parent_reaction_support_cells": [48, 64],
            "layouts": list(LAYOUTS),
            "forms": list(FORMS),
        },
    )
    _write(MANIFEST_PATH, manifest)
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Face-36 retained-mode Q+a pilot manifest\n\n"
        f"Classification: `{summary['classification']}`.\n\n"
        "This definitions-only package freezes a two-amplitude, output-oriented retained-memory architecture. The common basis captures at least "
        f"`{basis_metrics['minimum_form_energy_capture']:.6f}` of every declared middle/fine face-36 response form; its minimum layout-subspace cosine is "
        f"`{basis_metrics['minimum_layout_subspace_cosine']:.6f}`.\n\n"
        "The two coefficients are not yet physical modes or an oscillatory pair. The next package must construct state lifts, descriptor-weighted duals, and a ledger-derived macro-only reaction map B_Q, and must certify its KKT conditioning using existing states only.\n\n"
        "No fixed-Q microburst, nonlinear anchor, 50 ms trajectory, or reduced slow evolution is authorized. The raw face-48 export remains rejected, and the retained guard complement is not discarded.\n",
        encoding="utf-8",
    )
    _write(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "source_parent_commit": _read(CANONICAL_SUMMARY)[
                "latest_source_parent_commit"
            ],
            "execution_commit": _git("rev-parse", "HEAD"),
            "scientific_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
            "parent_summary_sha256": _sha(c4f13.SUMMARY_PATH),
            "parent_arrays_sha256": _sha(c4f13.DECISIVE_ARRAYS),
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST) if (ROOT / THIS_TEST).exists() else None,
            },
        },
    )
    files = (CONFIG_PATH, MANIFEST_PATH, MODE_BASIS_PATH, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
