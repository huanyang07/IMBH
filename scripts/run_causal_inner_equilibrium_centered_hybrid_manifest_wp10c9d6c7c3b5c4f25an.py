#!/usr/bin/env python3
"""Freeze the equilibrium-centered conservative slow-fast hybrid architecture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_intrinsic_constraint_geometry_audit_wp10c9d6c7c3b5c4f25am as geometry  # noqa: E402
import run_causal_inner_stable_parametric_online_audit_wp10c9d6c7c3b5c4f25ai as stable  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25an"
CLASSIFICATION = (
    "equilibrium_centered_conservative_slow_fast_hybrid_manifest_frozen_"
    "architecture_algebra_and_cost_audit_authorized"
)
PARENT_COMMIT = "67a64d0c79e458a2321f62079514c0ab3f51f836"
PARENT_PARENT = "63bee4f284530a9653fb5dd873854112107b2064"
PARENT_TREE = "a1166cf127c1dee7c456878bb68d72e36f88b3e9"

ARTIFACT = (
    "causal_inner_equilibrium_centered_hybrid_manifest_"
    "wp10c9d6c7c3b5c4f25an"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_equilibrium_centered_hybrid_manifest_"
    "wp10c9d6c7c3b5c4f25an.py"
)
THIS_TEST = (
    "tests/test_causal_inner_equilibrium_centered_hybrid_manifest_"
    "wp10c9d6c7c3b5c4f25an.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_equilibrium_centered_hybrid_architecture_audit_"
    "wp10c9d6c7c3b5c4f25ao.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_equilibrium_centered_hybrid_architecture_audit_"
    "wp10c9d6c7c3b5c4f25ao.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EQUILIBRIUM_CENTERED_HYBRID_"
    "MANIFEST_WP10C9D6C7C3B5C4F25AN_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

FULL_STATE_DIMENSION = 560
TRUE_CONSERVATIVE_DIMENSION = 96
CONSTITUTIVE_STORAGE_DIMENSION = 64
EXPLICIT_STABLE_DIMENSION = 2
RESOLVED_DIMENSION = 162
HIDDEN_DIMENSION = 398
STABLE_MEMORY_DIMENSION = 280
ELIMINATED_EVENT_DIMENSION = 28
TRUNCATED_STABLE_DIMENSION = 90
ONLINE_CONTINUOUS_DIMENSION = 442
CONSERVATIVE_COMPONENTS = 3
COARSE_CELLS = 32
FIDUCIAL_CYCLE_SECONDS = 6.7 * 86_400.0
WALL_BUDGET_SECONDS = 3.0 * 86_400.0
MAXIMUM_MACROSTEPS = 100_000
MINIMUM_AVERAGE_MACROSTEP_SECONDS = FIDUCIAL_CYCLE_SECONDS / MAXIMUM_MACROSTEPS


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


def _validate_parents() -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("hybrid-architecture parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("hybrid-architecture parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("hybrid-architecture parent tree changed")
    geometry_hashes = _checksums(geometry.CANONICAL_DIRECTORY)
    stable_hashes = _checksums(stable.CANONICAL_DIRECTORY)
    geometry_summary = _read(geometry.CANONICAL_DIRECTORY / "summary.json")
    stable_summary = _read(stable.CANONICAL_DIRECTORY / "summary.json")
    if (
        not geometry_summary["passed"]
        or geometry_summary["authorized_next"]
        != "definitions_only_constrained_equilibrium_branch_and_fast_transition_collocation_manifest"
        or geometry_summary["instantaneous_spectrum_is_normal_hyperbolicity_certificate"]
        or geometry_summary["selected_architecture"]
        != "equilibrium_centered_conservative_slow_fast_hybrid"
    ):
        raise RuntimeError("intrinsic geometry certificate changed")
    if (
        not stable_summary["passed"]
        or stable_summary["stable_descriptor_dimension"] != ONLINE_CONTINUOUS_DIMENSION
        or stable_summary["online_truth_calls_per_macrostep"] != 0
        or stable_summary["unstable_bundle_linear_macro_propagation_authorized"]
    ):
        raise RuntimeError("stable online-kernel certificate changed")
    return {
        "geometry_summary": geometry_summary,
        "stable_summary": stable_summary,
        "geometry_hashes": geometry_hashes,
        "stable_hashes": stable_hashes,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "architecture": "equilibrium_centered_conservative_slow_fast_hybrid",
        "state_partition": {
            "full_scaled_primitive_state": FULL_STATE_DIMENSION,
            "resolved_physical_observables": RESOLVED_DIMENSION,
            "resolved_true_conservative_M_J_E_storage": TRUE_CONSERVATIVE_DIMENSION,
            "resolved_constitutive_storage": CONSTITUTIVE_STORAGE_DIMENSION,
            "resolved_explicit_stable_coordinates": EXPLICIT_STABLE_DIMENSION,
            "hidden_full_order_fiber": HIDDEN_DIMENSION,
            "stable_memory_upper_bound": STABLE_MEMORY_DIMENSION,
            "event_eliminated_departure_bundle": ELIMINATED_EVENT_DIMENSION,
            "truncated_stable_remainder": TRUNCATED_STABLE_DIMENSION,
            "online_continuous_dimension_upper_bound": ONLINE_CONTINUOUS_DIMENSION,
            "online_discrete_state": "sigma_in_cold_hot_transition",
            "dimension_identities": [
                "560_equals_162_plus_398",
                "162_equals_96_plus_64_plus_2",
                "398_equals_280_plus_28_plus_90",
                "442_equals_162_plus_280",
            ],
        },
        "resolved_coordinate_map": {
            "map": "C_phys_x_equals_160_nonlinear_coarse_mapped_storage_observables_plus_2_explicit_stable_coordinates",
            "required_rank": RESOLVED_DIMENSION,
            "global_Q3_is_determined_by": "global_sums_of_the_96_M_J_E_storage_coordinates",
            "finite_amplitude_coordinates_are_physical_observables_not_a_linear_state_projection": True,
        },
        "conditional_fast_branch": {
            "unknown": "x_in_R560_at_fixed_resolved_y_in_R162_and_branch_sigma",
            "coordinate_equations": "C_phys_x_minus_y_equals_zero_in_R162",
            "hidden_basis": "H_x_spans_kernel_D_C_phys_x_and_has_398_columns",
            "fixed_Q_rate": "F_Q_x_is_the_exact_continuous_reaction_constrained_rate_with_DQ3_x_F_Q_x_equals_zero",
            "hidden_stationarity": "H_x_transpose_W_x_F_Q_x_equals_zero_in_R398",
            "square_equation_count": FULL_STATE_DIMENSION,
            "interpretation": "the_full_rate_may_be_nonzero_but_is_tangent_to_the_resolved_slow_manifold",
            "solver": "damped_complete_Newton_with_pseudo_arclength_continuation_along_a_frozen_low_dimensional_resolved_path",
            "multiple_roots": "cold_and_hot_branches_are_separate_solutions_at_the_same_resolved_coordinates",
            "moving_16ms_and_20ms_checkpoints_are_assumed_to_be_branch_roots": False,
        },
        "normal_hyperbolicity": {
            "operator": "A_fast_equals_derivative_with_respect_to_hidden_coordinates_of_H_transpose_W_F_Q_at_a_conditional_branch_root",
            "mass_metric": "G_fast_equals_H_transpose_W_H_is_SPD",
            "stable_branch_gate": "all_generalized_fast_eigenvalues_have_real_part_at_most_minus_gamma_min",
            "event_indicator": "loss_of_normal_hyperbolicity_or_independently_identified_basin_boundary",
            "instantaneous_checkpoint_spectrum_is_binding": False,
            "legacy_28_mode_spectrum_role": "diagnostic_seed_for_branch_and_event_search_only",
        },
        "online_continuous_dynamics": {
            "conservative_law": "dot_c_equals_minus_D_volume_inverse_Phi_sigma_c_eta_z_plus_S_c",
            "conservative_components": ["mass", "angular_momentum", "energy"],
            "finite_volume_cells": COARSE_CELLS,
            "constitutive_and_memory_descriptor": "G_sigma_dot_w_equals_K_sigma_w_plus_B_sigma_dot_c_plus_f_sigma_with_w_in_R346",
            "descriptor_dimension": (
                CONSTITUTIVE_STORAGE_DIMENSION
                + EXPLICIT_STABLE_DIMENSION
                + STABLE_MEMORY_DIMENSION
            ),
            "descriptor_conditions": "G_sigma_is_SPD_and_K_sigma_plus_K_sigma_transpose_is_negative_semidefinite_with_strict_decay_on_the_memory_subspace",
            "macro_integrator": "finite_volume_IMEX_for_c_coupled_to_exponential_or_L_stable_descriptor_update_for_w",
            "step_controller": "slow_flux_error_and_event_localization_not_the_fastest_stable_pole",
            "online_full_order_truth_calls_per_macrostep": 0,
            "linearly_macro_propagated_unstable_coordinates": 0,
        },
        "fast_transition_collocation": {
            "coordinates": "intrinsic_fixed_Q3_coordinates_of_dimension_557",
            "equation": "M_x_dx_ds_equals_T_transition_F_Q_x_on_s_in_0_1",
            "unknown_duration": True,
            "boundary_conditions": "incoming_and_outgoing_branch_sections_plus_one_phase_condition",
            "method": "adaptive_orthogonal_collocation_BVP_with_a_square_bordered_Jacobian_reported_by_the_discretizer",
            "state_history_is_synthesized_by_backward_projection": False,
            "transition_is_computed_by_tiny_forward_BDF_microsteps": False,
            "reset": "c_plus_minus_c_minus_equals_integral_of_finite_volume_flux_divergence_and_sources_through_the_transition",
            "global_ledger": "sum_c_plus_minus_sum_c_minus_equals_integrated_boundary_and_source_impulse",
            "global_Q3_preserved_without_external_impulse": True,
        },
        "offline_database": {
            "initial_training_path_anchors": 12,
            "sealed_heldout_path_anchors": 6,
            "maximum_adaptive_training_anchors": 30,
            "heldout_data_may_change_basis_events_or_tolerances": False,
            "required_objects": [
                "cold_and_hot_conditional_branch_states",
                "equilibrium_centered_fast_spectra",
                "conservative_face_flux_closure",
                "stable_memory_descriptor_pairs",
                "separate_up_and_down_event_surfaces",
                "collocated_transition_orbits",
                "conservative_reset_and_impulse_maps",
            ],
            "branch_or_transition_existence_assumed": False,
        },
        "runtime_contract": {
            "fiducial_cycle_seconds": FIDUCIAL_CYCLE_SECONDS,
            "wall_budget_seconds": WALL_BUDGET_SECONDS,
            "maximum_macrosteps": MAXIMUM_MACROSTEPS,
            "minimum_average_macrostep_seconds": MINIMUM_AVERAGE_MACROSTEP_SECONDS,
            "stable_kernel_budget_seconds": 0.10 * WALL_BUDGET_SECONDS,
            "all_online_algebra_budget_seconds": 0.50 * WALL_BUDGET_SECONDS,
            "complete_cycle_budget_seconds": WALL_BUDGET_SECONDS,
            "maximum_expected_branch_events_per_cycle": 4,
            "offline_truth_cost_is_not_charged_to_online_cycle_runtime": True,
        },
        "binding_architecture_audit_gates": {
            "dimension_arithmetic_exact": True,
            "finite_volume_global_telescoping_defect_max": 1.0e-14,
            "minimum_norm_reset_constraint_defect_max": 1.0e-12,
            "legacy_descriptor_energy_amplification_max": 1.0,
            "legacy_stable_kernel_projected_cycle_wall_seconds_max": 0.10
            * WALL_BUDGET_SECONDS,
            "online_continuous_dimension_max": ONLINE_CONTINUOUS_DIMENSION,
            "online_truth_calls_per_macrostep_equal": 0,
            "conditional_branch_unknown_count_equal": FULL_STATE_DIMENSION,
            "conditional_branch_equation_count_equal": FULL_STATE_DIMENSION,
        },
        "decision": {
            "pass": "equilibrium_centered_conservative_slow_fast_hybrid_architecture_certified_offline_branch_seed_manifest_authorized",
            "fail": "hybrid_architecture_structural_audit_failed_reduced_slow_evolution_blocked",
            "pass_authorizes_only": "definitions_only_first_conditional_fast_branch_seed_manifest",
        },
        "claim_boundary": {
            "physical_conditional_branch_found": False,
            "cold_hot_branch_pair_found": False,
            "fast_transition_orbit_found": False,
            "equilibrium_centered_closure_coefficients_identified": False,
            "legacy_442_coefficients_promoted_to_branch_closure": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
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
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
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


def _freeze() -> dict:
    parents = _validate_parents()
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("hybrid architecture manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("hybrid architecture manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "selected_architecture": "equilibrium_centered_conservative_slow_fast_hybrid",
        "resolved_dimension": RESOLVED_DIMENSION,
        "hidden_dimension": HIDDEN_DIMENSION,
        "online_continuous_dimension_upper_bound": ONLINE_CONTINUOUS_DIMENSION,
        "online_truth_calls_per_macrostep": 0,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25ao",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "intrinsic_geometry_package_hashes": parents["geometry_hashes"],
            "stable_online_package_hashes": parents["stable_hashes"],
        },
    )
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
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
            "thread_environment": stable.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Equilibrium-centered conservative slow-fast hybrid manifest WP10c9d6c7c3b5c4f25an",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The selected architecture uses 162 physical resolved observables and solves a square 398-dimensional hidden stationarity problem at fixed resolved state. The resulting conditional branches may carry a nonzero resolved slow rate; the moving 16 ms and 20 ms checkpoints are not assumed to be equilibria.",
                "",
                "Online evolution retains 96 exactly conservative finite-volume coordinates, 66 constitutive/stable coordinates, at most 280 stable memory coordinates, and a discrete cold/hot/transition label. The 28 positive-growth directions are removed from linear macro propagation and represented through branch events and conservative collocated resets.",
                "",
                "A structural audit is authorized. No physical branch root, transition, online solver, or predictive cycle is authorized by this manifest.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
