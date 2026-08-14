#!/usr/bin/env python3
"""Freeze the six-mode face-36 dynamic-coordinate preflight contract.

Definitions only.  This package neither advances a tangent/nonlinear
trajectory nor applies a fixed-Q reaction.
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
import run_causal_inner_face36_q_plus_a_reaction_coordinate_preflight_wp10c9d6c7c3b5c4f15 as c4f15  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f16"
ARTIFACT = "causal_inner_face36_six_mode_coordinate_manifest_wp10c9d6c7c3b5c4f16"
THIS_RUNNER = "scripts/run_causal_inner_face36_six_mode_coordinate_manifest_wp10c9d6c7c3b5c4f16.py"
THIS_TEST = "tests/test_causal_inner_face36_six_mode_coordinate_manifest_wp10c9d6c7c3b5c4f16.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_FACE36_SIX_MODE_COORDINATE_MANIFEST_WP10C9D6C7C3B5C4F16_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "dynamic_coordinate_manifest.json"
BASIS_PATH = CANONICAL_DIRECTORY / "six_mode_basis.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

LAYOUTS = ("middle", "fine")
FORMS = ("instantaneous", "cumulative", "window_mean")
MODE_DIMENSION = 6
LEADING_BLOCK_DIMENSION = 2


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
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
    summary = _read(c4f15.SUMMARY_PATH)
    if (
        not summary["audit_completed"]
        or summary["passed"]
        or not summary["reaction_map_preflight_passed"]
        or not summary["endpoint_coordinate_preflight_passed"]
        or summary["two_mode_significant_direction_gate_passed"]
        or summary["minimum_passing_output_oriented_dimension"] != MODE_DIMENSION
        or summary["fixed_Q_micro_solver_authorized"]
        or summary["nonlinear_retained_mode_pilot_authorized"]
        or summary["authorized_next"]
        != "WP10c9d6c7c3b5c4f16_definitions_only_six_mode_Q_plus_a_coordinate_manifest"
    ):
        raise RuntimeError("c4f16 authorization changed")
    return summary


def _normalized_gram(singular: np.ndarray, right: np.ndarray) -> np.ndarray:
    gram = (right.T * np.square(singular)) @ right
    return gram / np.trace(gram)


def _basis_diagnostics():
    with np.load(c4f13.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        layout_grams = {}
        form_grams = {}
        for label in LAYOUTS:
            total = np.zeros((c4f13.TOTAL_DIRECTIONS,) * 2)
            for form in FORMS:
                gram = _normalized_gram(
                    arrays[f"{label}__{form}_singular_values"],
                    arrays[f"{label}__{form}_right_vectors"],
                )
                form_grams[(label, form)] = gram
                total += gram / len(FORMS)
            layout_grams[label] = total
    consensus = sum(layout_grams.values()) / len(LAYOUTS)

    def eigensystem(matrix):
        values, vectors = np.linalg.eigh(matrix)
        order = np.argsort(values)[::-1]
        return values[order], vectors[:, order]

    values, vectors = eigensystem(consensus)
    middle_values, middle_vectors = eigensystem(layout_grams["middle"])
    fine_values, fine_vectors = eigensystem(layout_grams["fine"])
    basis = vectors[:, :MODE_DIMENSION]
    for mode in range(MODE_DIMENSION):
        pivot = int(np.argmax(np.abs(basis[:, mode])))
        if basis[pivot, mode] < 0.0:
            basis[:, mode] *= -1.0
    middle_basis = middle_vectors[:, :MODE_DIMENSION]
    fine_basis = fine_vectors[:, :MODE_DIMENSION]
    leading_middle = middle_vectors[:, :LEADING_BLOCK_DIMENSION]
    leading_fine = fine_vectors[:, :LEADING_BLOCK_DIMENSION]
    form_capture = {
        label: {
            form: float(
                np.trace(basis.T @ form_grams[(label, form)] @ basis)
            )
            for form in FORMS
        }
        for label in LAYOUTS
    }
    full_cosines = np.linalg.svd(
        middle_basis.T @ fine_basis, compute_uv=False
    )
    leading_cosines = np.linalg.svd(
        leading_middle.T @ leading_fine, compute_uv=False
    )
    return {
        "six_mode_consensus_energy_capture": float(
            np.sum(values[:MODE_DIMENSION]) / np.sum(values)
        ),
        "minimum_six_mode_form_capture": min(
            min(values.values()) for values in form_capture.values()
        ),
        "sigma6_over_sigma7_gap": float(values[5] / values[6]),
        "middle_fine_leading_block_principal_cosines": leading_cosines.tolist(),
        "minimum_middle_fine_leading_block_cosine": float(
            np.min(leading_cosines)
        ),
        "middle_fine_six_mode_principal_cosines": full_cosines.tolist(),
        "minimum_middle_fine_six_mode_cosine": float(np.min(full_cosines)),
        "form_capture": form_capture,
    }, {
        "six_mode_consensus_direction_coefficients": basis,
        "consensus_gram_eigenvalues": values,
        "middle_gram_eigenvalues": middle_values,
        "fine_gram_eigenvalues": fine_values,
        "middle_six_mode_local_basis": middle_basis,
        "fine_six_mode_local_basis": fine_basis,
    }


def _manifest(parent: dict, metrics: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "face36_six_mode_output_closure_manifest_frozen_"
            "dynamic_coordinate_preflight_authorized"
        ),
        "definitions_only": True,
        "new_trajectory": False,
        "physical_operator_changed": False,
        "parent_decision": {
            "reaction_map_preflight_passed": True,
            "two_mode_coordinate_rejected": True,
            "minimum_output_closing_dimension": MODE_DIMENSION,
            "six_mode_output_metrics": parent[
                "minimum_passing_dimension_output_reconstruction"
            ],
        },
        "coordinate_architecture": {
            "state": "Q_M_J_E_plus_six_output_oriented_amplitudes",
            "leading_block": {
                "dimensions": [0, 2],
                "interpretation": "cross_grid_stable_dominant_output_subspace",
            },
            "weak_enrichment_block": {
                "dimensions": [2, 6],
                "interpretation": (
                    "rotatable_low_energy_closure_enrichment_subspace_not_"
                    "four_individually_identified_physical_modes"
                ),
            },
            "consensus_basis_metrics": metrics,
            "individual_mode_matching_for_dimensions_3_to_6_forbidden": True,
            "Procrustes_or_projector_comparison_required": True,
            "eigenmode_or_oscillatory_pair_claim_forbidden": True,
            "guard_state_retained": True,
        },
        "reaction_contract": {
            "reuse_c4f15_ledger_derived_B_Q": True,
            "support_parent_cells": [48, 64],
            "macro_Q_parent_cells": [36, 64],
            "shared_exchange_parent_face": 36,
            "raw_face48_exchange_forbidden": True,
            "no_reaction_applied_in_dynamic_coordinate_preflight": True,
            "reason": (
                "the_next_preflight_is_a_kinematic_coordinate_replay_not_a_"
                "fixed_Q_attractor_experiment"
            ),
        },
        "authorized_dynamic_coordinate_preflight": {
            "work_package": (
                "WP10c9d6c7c3b5c4f17_analysis_only_six_mode_"
                "dynamic_coordinate_replay"
            ),
            "uses_committed_middle_and_fine_5_to_20ms_base_histories": True,
            "new_nonlinear_trajectory": False,
            "directions": MODE_DIMENSION,
            "initial_state_lifts": (
                "frozen_29_Q3_null_lifts_times_six_mode_consensus_coefficients"
            ),
            "dual": (
                "fixed_5ms_descriptor_weighted_constraint_compatible_"
                "Petrov_dual"
            ),
            "amplitude_transition_history": "A_5ms_times_delta_p_mode_of_t",
            "propagate_complete_BDF_history_directions": True,
            "save_state_direction_history_at_all_committed_outputs": True,
            "save_guard_mapped_and_height_history_complement": True,
            "compare_middle_then_fine_after_common_parent_restriction": True,
            "align_weak_block_by_orthogonal_Procrustes": True,
            "run_middle_first": True,
            "run_fine_only_after_middle_method_and_coordinate_gates_pass": True,
            "independent_complete_residual_JVP_only_at_times_ms": [
                5.4, 10.0, 16.0, 20.0
            ],
        },
        "prospective_dynamic_gates": {
            "maximum_step_matrix_JVP_relative_defect": 1.0e-8,
            "maximum_block_linear_solve_relative_defect": 1.0e-10,
            "maximum_face36_output_map_relative_defect": 1.0e-8,
            "maximum_Q3_leakage": 0.10,
            "maximum_initial_state_lift_Q3_defect": 1.0e-10,
            "maximum_dual_biorthogonality_defect": 1.0e-10,
            "maximum_normalized_slow_lift_annihilation_defect": 1.0e-10,
            "minimum_middle_fine_leading_block_projector_cosine": 0.95,
            "minimum_middle_fine_full_subspace_projector_cosine": 0.90,
            "minimum_middle_fine_amplitude_transition_history_cosine": 0.95,
            "maximum_middle_fine_amplitude_transition_relative_difference": 0.10,
            "minimum_middle_fine_face36_mode_history_cosine": 0.95,
            "maximum_middle_fine_face36_mode_history_relative_difference": 0.10,
            "maximum_six_mode_output_weighted_RMS_error": 0.10,
            "maximum_six_mode_significant_direction_error": 0.25,
            "incoming_excision_characteristics": 0,
        },
        "fail_fast_decision": {
            "middle_method_fails": "stop_before_fine",
            "leading_block_fails": "return_to_memory_basis_localization",
            "weak_block_only_fails": (
                "retain_two_stable_coordinates_and_represent_residual_"
                "closure_with_HMM_or_guard_microstate"
            ),
            "six_mode_dynamic_preflight_passes": (
                "authorize_definitions_only_one_Q_constrained_pilot_manifest"
            ),
        },
        "cost_contract": {
            "measured_parent_middle_wall_hours": 4661.505936835427 / 3600.0,
            "measured_parent_fine_wall_hours": 10268.69857237162 / 3600.0,
            "expected_total_wall_hours": [3.5, 5.0],
            "one_factorization_six_RHS_per_step": True,
            "block_solve_marginal_cost_negligible": True,
            "no_repeated_29_direction_propagation": True,
            "durable_middle_and_fine_checkpoints": True,
        },
        "hard_stops": [
            "do_not_call_weak_enrichment_vectors_individual_physical_modes",
            "do_not_apply_Euclidean_fixed_Q_projection",
            "do_not_start_a_fixed_Q_or_nonlinear_microburst",
            "do_not_discard_the_guard_complement",
            "do_not_use_raw_face48_as_slow_exchange",
            "do_not_start_50ms_or_reduced_slow_evolution",
        ],
        "dynamic_coordinate_preflight_authorized": True,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f17_analysis_only_six_mode_"
            "dynamic_coordinate_replay"
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
    metrics, stored = _basis_diagnostics()
    manifest = _manifest(parent, metrics)
    if (
        parent["minimum_passing_dimension_output_reconstruction"][
            "maximum_output_weighted_RMS_error"
        ] > 0.10
        or parent["minimum_passing_dimension_output_reconstruction"][
            "maximum_significant_direction_error"
        ] > 0.25
        or metrics["minimum_middle_fine_leading_block_cosine"] < 0.95
    ):
        raise RuntimeError("c4f16 six-mode or leading-block gate changed")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "mode_dimension": MODE_DIMENSION,
        "basis_metrics": metrics,
        "six_mode_output_closure_passed": True,
        "six_mode_dynamic_coordinate_certified": False,
        "weak_enrichment_individual_mode_identity_rejected": True,
        "dynamic_coordinate_preflight_authorized": True,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "guard_complement_retained": True,
        "raw_face48_export_rejection_preserved": True,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BASIS_PATH, **stored)
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "mode_dimension": MODE_DIMENSION,
            "leading_block_dimension": LEADING_BLOCK_DIMENSION,
            "layouts": list(LAYOUTS),
            "forms": list(FORMS),
            "parent_macro_cells": [36, 64],
            "parent_guard_cells": [36, 48],
            "parent_reaction_support_cells": [48, 64],
            "shared_exchange_parent_face": 36,
        },
    )
    _write(MANIFEST_PATH, manifest)
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Face-36 six-mode coordinate manifest\n\n"
        f"Classification: `{summary['classification']}`.\n\n"
        "This definitions-only package preserves the successful c4f15 physical reaction map and replaces the rejected two-mode coordinate candidate with a six-dimensional output-closing candidate. The six-mode basis captures at least "
        f"`{metrics['minimum_six_mode_form_capture']:.8f}` of each declared output Gramian and has a mode-6/mode-7 gap of `{metrics['sigma6_over_sigma7_gap']:.6f}`.\n\n"
        "Only the leading two-dimensional block is presently cross-grid stable. The minimum middle/fine principal cosine of the full local six-dimensional spaces is "
        f"`{metrics['minimum_middle_fine_six_mode_cosine']:.6f}`. Dimensions three through six are therefore frozen as one rotatable weak enrichment block, not as four named physical modes.\n\n"
        "The next package may replay six tangent directions through the committed middle and fine 5--20 ms histories, middle first. It must save genuine state-direction histories and compare common-parent state projectors, fixed-dual amplitude-transition histories, face-36 mode histories, and the retained guard complement. No fixed-Q reaction, nonlinear microburst, 50 ms run, or reduced slow evolution is authorized.\n",
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
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "parent_summary_sha256": _sha(c4f15.SUMMARY_PATH),
            "parent_arrays_sha256": _sha(c4f15.DECISIVE_ARRAYS),
            "memory_arrays_sha256": _sha(c4f13.DECISIVE_ARRAYS),
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST) if (ROOT / THIS_TEST).exists() else None,
            },
        },
    )
    files = (CONFIG_PATH, MANIFEST_PATH, BASIS_PATH, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
