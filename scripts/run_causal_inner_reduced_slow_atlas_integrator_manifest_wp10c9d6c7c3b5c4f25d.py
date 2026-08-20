#!/usr/bin/env python3
"""Freeze the reduced slow-atlas mathematical architecture."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_forward_quadratic_field_blind_validation_wp10c9d6c7c3b5c4f25cz as parent  # noqa: E402
import run_causal_inner_reduced_cycle_identifiability_wp10c9d6c7c3b5c4f25a as prior_architecture  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25d"
PARENT_COMMIT = "3cce7dd77aef260955b107aa6cb0565e9f8ed637"
PARENT_PARENT = "b5c770d8362934543e1af5cb5ffd0c4ba9307e9f"
PARENT_TREE = "10f3e7ff22a936e33b1363f12c0d118c0dfad142"

CLASSIFICATION = (
    "reduced_slow_atlas_integrator_architecture_frozen_"
    "local_slaving_preflight_authorized"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25da"

FIDUCIAL_CYCLE_SECONDS = 6.7 * 86400.0
WALL_BUDGET_SECONDS = 3.0 * 86400.0
MAXIMUM_MACROSTEPS_PER_CYCLE = 100_000
REFERENCE_MICROSTEP_SECONDS = 1.0e-7
REFERENCE_RHS_EVALUATIONS_PER_MACROSTEP = 8

ARTIFACT = (
    "causal_inner_reduced_slow_atlas_integrator_manifest_"
    "wp10c9d6c7c3b5c4f25d"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_reduced_slow_atlas_integrator_manifest_"
    "wp10c9d6c7c3b5c4f25d.py"
)
THIS_TEST = (
    "tests/test_causal_inner_reduced_slow_atlas_integrator_manifest_"
    "wp10c9d6c7c3b5c4f25d.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_REDUCED_SLOW_ATLAS_"
    "INTEGRATOR_MANIFEST_WP10C9D6C7C3B5C4F25D_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

VALIDATION_ARRAYS = parent.CANONICAL_DIRECTORY / "validation_arrays.npz"
FIELD_ARRAYS = parent.FROZEN_FIELD
GEOMETRY_ARRAYS = parent.GEOMETRY_ARRAYS
PRIOR_ARCHITECTURE_DIRECTORY = prior_architecture.ARTIFACT_DIRECTORY

field_manifest = parent.field_manifest

_plain = field_manifest._plain
_read = field_manifest._read
_write_json = field_manifest._write_json
_sha = field_manifest._sha
_checksums = field_manifest._checksums
_load_npz = field_manifest._load_npz


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("forward-quadratic field certificate commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("forward-quadratic field certificate lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("forward-quadratic field certificate tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "validation_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    arrays = _load_npz(VALIDATION_ARRAYS)
    field_hashes = _checksums(field_manifest.CANONICAL_DIRECTORY)
    field = _load_npz(FIELD_ARRAYS)
    geometry = _load_npz(GEOMETRY_ARRAYS)
    prior_hashes = prior_architecture._checksums(PRIOR_ARCHITECTURE_DIRECTORY)
    prior_summary = _read(PRIOR_ARCHITECTURE_DIRECTORY / "summary.json")
    prior_decision = _read(PRIOR_ARCHITECTURE_DIRECTORY / "decision.json")
    if (
        not summary["passed"]
        or summary["classification"] != parent.PASS_CLASSIFICATION
        or summary["authorized_next"]
        != "definitions_only_reduced_slow_atlas_integrator_manifest"
        or summary["completed_exact_rate_calls"] != 4
        or summary["failed_exact_rate_calls"] != 0
        or not summary["blind_holdout_passed"]
        or summary["coefficients_refit_after_holdout_truth"]
        or summary["truth_calls_repeated_after_postprocessing_repair"] != 0
        or not metrics["passed"]
        or not all(metrics["checks"].values())
        or arrays["total_rates_per_second"].shape != (4, 560)
        or arrays["exact_coordinate_rates_per_second"].shape != (4, 470)
        or field["revealed_exact_full_rates_per_second"].shape != (9, 560)
        or geometry["candidate_active_coordinates"].shape != (4, 3)
        or not prior_summary["passed"]
        or prior_summary["selected_architecture"]
        != "cellwise_Q5_FV_plus_a2_finite_memory_hybrid"
        or not prior_summary["offline_closure_database_manifest_authorized"]
        or prior_summary["online_reduced_solver_implementation_authorized"]
        or prior_decision["coefficients_identifiable_from_existing_committed_data"]
    ):
        raise RuntimeError("reduced slow-atlas authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"validated field source changed: {relative}")
    if (
        _sha(VALIDATION_ARRAYS) != hashes["validation_arrays.npz"]
        or _sha(FIELD_ARRAYS) != field_hashes["forward_quadratic_local_field.npz"]
    ):
        raise RuntimeError("reduced slow-atlas input changed")
    prior_architecture._checksums(PRIOR_ARCHITECTURE_DIRECTORY)
    for name, expected in field_manifest.training._thread_environment().items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("reduced slow-atlas manifest requires a clean tracked tree")
    return {
        "summary": summary,
        "metrics": metrics,
        "hashes": hashes,
        "arrays": arrays,
        "field_hashes": field_hashes,
        "field": field,
        "geometry": geometry,
        "prior_hashes": prior_hashes,
        "prior_summary": prior_summary,
        "prior_decision": prior_decision,
    }


def _seed_database(frozen: dict) -> tuple[dict[str, np.ndarray], dict]:
    field = frozen["field"]
    validation = frozen["arrays"]
    geometry = frozen["geometry"]
    local = np.vstack(
        (field["revealed_local_coordinates"], validation["holdout_local_coordinates"])
    )
    active = np.vstack(
        (field["revealed_active_coordinates"], geometry["candidate_active_coordinates"])
    )
    exact_full = np.vstack(
        (field["revealed_exact_full_rates_per_second"], validation["total_rates_per_second"])
    )
    predicted_full = np.vstack(
        (field["revealed_predicted_full_rates_per_second"], validation["predicted_full_rates_per_second"])
    )
    exact_coordinate = np.vstack(
        (field["revealed_exact_coordinate_rates_per_second"], validation["exact_coordinate_rates_per_second"])
    )
    predicted_coordinate = np.vstack(
        (field["revealed_predicted_coordinate_rates_per_second"], validation["predicted_coordinate_rates_per_second"])
    )
    exact_jacobian = np.vstack(
        (field["revealed_exact_q162_Jacobians"], validation["exact_q162_Jacobians"])
    )
    predicted_jacobian = np.vstack(
        (field["revealed_predicted_q162_Jacobians"], validation["predicted_q162_Jacobians"])
    )
    if (
        local.shape != (13, 470)
        or active.shape != (13, 3)
        or exact_full.shape != (13, 560)
        or exact_coordinate.shape != (13, 470)
        or exact_jacobian.shape != (13, 162, 560)
    ):
        raise RuntimeError("slow-atlas seed database changed")
    centered = active - np.mean(active, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    full_errors = np.linalg.norm(predicted_full - exact_full, axis=1) / np.maximum(
        np.linalg.norm(exact_full, axis=1), np.finfo(float).tiny
    )
    coordinate_errors = np.linalg.norm(
        predicted_coordinate - exact_coordinate, axis=1
    ) / np.maximum(np.linalg.norm(exact_coordinate, axis=1), np.finfo(float).tiny)
    q_errors = np.linalg.norm(
        predicted_coordinate[:, :162] - exact_coordinate[:, :162], axis=1
    ) / np.maximum(
        np.linalg.norm(exact_coordinate[:, :162], axis=1), np.finfo(float).tiny
    )
    jacobian_errors = np.linalg.norm(
        (predicted_jacobian - exact_jacobian).reshape(13, -1), axis=1
    ) / np.maximum(
        np.linalg.norm(exact_jacobian.reshape(13, -1), axis=1),
        np.finfo(float).tiny,
    )
    metrics = {
        "seed_exact_rate_count": 13,
        "seed_active_dimension": 3,
        "seed_active_affine_rank": int(np.linalg.matrix_rank(np.c_[np.ones(13), active])),
        "minimum_centered_active_singular_value": float(singular[-1]),
        "maximum_seed_full_state_rate_relative_error": float(np.max(full_errors)),
        "maximum_seed_full_coordinate_rate_relative_error": float(
            np.max(coordinate_errors)
        ),
        "maximum_seed_q162_rate_relative_error": float(np.max(q_errors)),
        "maximum_seed_q162_Jacobian_relative_error": float(
            np.max(jacobian_errors)
        ),
        "minimum_active_forward_coordinate": float(np.min(active[:, 0])),
        "maximum_active_forward_coordinate": float(np.max(active[:, 0])),
        "maximum_active_transverse_radius": float(
            np.max(np.linalg.norm(active[:, 1:], axis=1))
        ),
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
    }
    arrays = {
        "seed_local_coordinates": local,
        "seed_active_coordinates": active,
        "seed_exact_full_rates_per_second": exact_full,
        "seed_predicted_full_rates_per_second": predicted_full,
        "seed_exact_coordinate_rates_per_second": exact_coordinate,
        "seed_predicted_coordinate_rates_per_second": predicted_coordinate,
        "seed_exact_q162_Jacobians": exact_jacobian,
        "seed_predicted_q162_Jacobians": predicted_jacobian,
        "seed_full_state_rate_relative_errors": full_errors,
        "seed_full_coordinate_rate_relative_errors": coordinate_errors,
        "seed_q162_rate_relative_errors": q_errors,
        "seed_q162_Jacobian_relative_errors": jacobian_errors,
        "authentic_center_absolute_coordinate": field[
            "authentic_center_absolute_coordinate"
        ],
        "active_departure_basis": field["active_departure_basis"],
    }
    return arrays, metrics


def _runtime_budget(frozen: dict) -> dict:
    rhs_seconds = float(frozen["metrics"]["median_online_field_wall_seconds"])
    direct_steps = FIDUCIAL_CYCLE_SECONDS / REFERENCE_MICROSTEP_SECONDS
    direct_surrogate_wall_seconds = direct_steps * rhs_seconds
    macrostep_seconds = FIDUCIAL_CYCLE_SECONDS / MAXIMUM_MACROSTEPS_PER_CYCLE
    reference_macro_wall_seconds = (
        MAXIMUM_MACROSTEPS_PER_CYCLE
        * REFERENCE_RHS_EVALUATIONS_PER_MACROSTEP
        * rhs_seconds
    )
    return {
        "fiducial_cycle_seconds": FIDUCIAL_CYCLE_SECONDS,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "maximum_macrosteps_per_cycle": MAXIMUM_MACROSTEPS_PER_CYCLE,
        "minimum_average_macrostep_seconds": macrostep_seconds,
        "validated_field_median_rhs_wall_seconds": rhs_seconds,
        "reference_microstep_seconds": REFERENCE_MICROSTEP_SECONDS,
        "direct_surrogate_microsteps_per_cycle": direct_steps,
        "direct_surrogate_wall_years_per_cycle": direct_surrogate_wall_seconds
        / (365.25 * 86400.0),
        "reference_rhs_evaluations_per_macrostep": REFERENCE_RHS_EVALUATIONS_PER_MACROSTEP,
        "reference_100k_macrostep_rhs_wall_seconds": reference_macro_wall_seconds,
        "reference_100k_macrostep_rhs_wall_fraction": reference_macro_wall_seconds
        / WALL_BUDGET_SECONDS,
        "required_architectural_change": "eliminate_fast_stability_scale_online",
    }


def _architecture_contract(runtime: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "scientific_status": {
            "forward_quadratic_local_field_independently_validated": True,
            "local_field_is_direct_cycle_integrator": False,
            "physical_failure_detected": False,
            "remaining_bottleneck": "timescale_elimination_and_pathwise_domain_coverage",
        },
        "three_layer_architecture": {
            "offline_truth_layer": {
                "state": "primitive_560_with_exact_fixed_Q_rate_and_complete_Jacobian",
                "role": "sparse_anchor_and_heldout_validation_only",
                "online_calls": 0,
            },
            "offline_atlas_layer": {
                "state": "y470_equal_q162_z280_a28",
                "field": "C2_partitioned_forward_quadratic_validated_local_surrogate",
                "role": "local_slaving_memory_and_flux_identification",
                "direct_cycle_integration_forbidden": True,
            },
            "online_macro_layer": {
                "state": "cellwise_Q5_on_16_radial_cells_plus_a2_plus_stable_memory_r_plus_branch",
                "continuous_dimension_by_memory_order": {
                    "0": 82,
                    "2": 84,
                    "4": 86,
                    "6": 88,
                },
                "memory_orders_to_screen": [0, 2, 4, 6],
                "branch": ["cold", "hot", "transition"],
                "online_truth_calls": 0,
                "online_atlas_microbursts": 0,
            },
        },
        "slow_graph_and_memory": {
            "macro_state_symbol": "X=(U80,a2,m_r,b)",
            "lifting": "y=Psi_b(X)",
            "projected_invariance_equation": "D_Psi_b(X)G_b(X)=F_atlas(Psi_b(X))+controlled_residual",
            "fast_block_requirement": "stable_real_Schur_block",
            "minimum_spectral_gap_ratio": 10.0,
            "maximum_relative_invariance_defect": 1.0e-1,
            "memory_model": "stable_passive_rational_Mori_Zwanzig_kernel",
            "memory_equation": "dm/dt=-T_b_inverse_m+B_b(U,a)dU/dt",
            "stable_poles_required": True,
            "passivity_or_declared_dissipation_required": True,
            "fallback": "increase_conservative_macro_state_before_increasing_memory_above_6",
        },
        "conservative_macro_dynamics": {
            "cell_balance": "dU_i/dt=Phi_(i-1/2)-Phi_(i+1/2)+S_i-W_i",
            "single_valued_interior_face_flux": True,
            "telescoping_mass_angular_momentum_killing_energy_ledgers": True,
            "fit_face_fluxes_not_independent_cell_derivatives": True,
            "constraint_projection": "weighted_minimum_norm_flux_source_correction",
            "inner_boundary": "certified_face36_exterior_partition",
            "raw_horizon_face_flux_forbidden": True,
            "responsive_height_one_form_is_not_an_absolute_coordinate": True,
        },
        "atlas_and_domain_guard": {
            "current_exact_seed_count": 13,
            "current_patch_role": "local_forward_sector_seed_only",
            "global_forward_half_space_claimed": False,
            "trust_domain": "union_of_validated_sample_hulls_with_decoder_physical_and_error_guards",
            "unbounded_polynomial_extrapolation_forbidden": True,
            "leave_trust_domain_action": "stop_and_request_offline_patch_expansion",
            "patch_expansion": "pathwise_active_learning_with_coefficients_and_holdouts_frozen_before_truth",
            "global_tensor_product_grid_forbidden": True,
            "middle_layout_training": True,
            "sparse_fine_layout_validation": True,
        },
        "online_integrator": {
            "family": "second_order_one_step_conservative_IMEX_ARK",
            "linear_memory_update": "exact_exponential_or_L_stable_stage",
            "adaptive_error_estimator": "embedded_first_order_and_step_doubling_controls",
            "event_location": "bracketed_hysteresis_surface_root",
            "event_conservation": "U80_continuous_and_auxiliary_reset_ledger_exact",
            "accepted_step_guards": [
                "atlas_trust_domain",
                "storage_positivity",
                "height",
                "optical_depth",
                "mass_angular_momentum_energy_ledgers",
                "memory_dissipation",
                "branch_hysteresis",
            ],
            "rejected_step_never_propagated": True,
            "restart_after_every_accepted_macrostep": True,
        },
        "runtime_budget": runtime,
        "immediate_local_slaving_preflight": {
            "work_package": AUTHORIZED_NEXT,
            "new_exact_rate_calls_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_fixed_Q_roots_equal": 0,
            "propagated_physical_truth_states_equal": 0,
            "tasks": [
                "construct_16_cell_Q5_conservative_restriction_and_lifting",
                "audit_exact_global_M_J_E_telescoping_rows",
                "differentiate_validated_atlas_field_and_macro_map_on_seed_cloud",
                "compute_resolved_unresolved_real_Schur_split",
                "screen_stable_memory_orders_0_2_4_6",
                "measure_slow_graph_invariance_and_transfer_fit_by_leave_one_seed_out",
                "freeze_new_independent_patch_holdouts_before_any_future_truth",
            ],
            "binding_gates": {
                "maximum_global_conservative_projection_defect": 1.0e-12,
                "minimum_macro_restriction_rank": 80,
                "maximum_field_Jacobian_step_ladder_relative_defect": 1.0e-3,
                "maximum_fast_spectral_abscissa_per_second": 0.0,
                "minimum_spectral_gap_ratio": 10.0,
                "maximum_relative_invariance_defect": 1.0e-1,
                "maximum_leave_one_seed_out_projected_rate_relative_error": 7.5e-2,
                "maximum_memory_transfer_relative_error": 1.0e-1,
                "stable_memory_poles_required": True,
            },
        },
        "decision": {
            "pass_classification": "local_slow_graph_and_finite_memory_architecture_supported",
            "pass_authorizes_only": "definitions_only_pathwise_slow_atlas_expansion_manifest",
            "unstable_fast_block_classification": "local_slow_reduction_rejected_unstable_fast_block",
            "no_gap_classification": "compact_slow_graph_rejected_expand_conservative_macro_state",
            "memory_fit_failure_classification": "finite_memory_order_six_insufficient_expand_conservative_macro_state",
        },
        "authorization_boundaries": {
            "online_reduced_solver_implementation_authorized": False,
            "physical_microburst_authorized": False,
            "exploratory_cycle_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _checks(seed_metrics: dict, runtime: dict) -> dict[str, bool]:
    return {
        "seed_count": seed_metrics["seed_exact_rate_count"] == 13,
        "seed_active_rank": seed_metrics["seed_active_affine_rank"] == 4,
        "seed_full_state_error": seed_metrics[
            "maximum_seed_full_state_rate_relative_error"
        ]
        <= 7.5e-2,
        "seed_full_coordinate_error": seed_metrics[
            "maximum_seed_full_coordinate_rate_relative_error"
        ]
        <= 7.5e-2,
        "seed_q162_error": seed_metrics["maximum_seed_q162_rate_relative_error"]
        <= 7.5e-2,
        "seed_q162_Jacobian_error": seed_metrics[
            "maximum_seed_q162_Jacobian_relative_error"
        ]
        <= 5.0e-3,
        "macrostep_is_cycle_scale": runtime["minimum_average_macrostep_seconds"]
        >= 1.0,
        "reference_rhs_budget": runtime[
            "reference_100k_macrostep_rhs_wall_fraction"
        ]
        <= 0.10,
        "truth_budget": seed_metrics["new_exact_rate_calls"] == 0,
        "generator_budget": seed_metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": seed_metrics["new_nonlinear_fixed_Q_roots"] == 0,
        "propagation_budget": seed_metrics["propagated_states"] == 0,
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
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("reduced slow-atlas manifest already exists")
    seed_arrays, seed_metrics = _seed_database(frozen)
    runtime = _runtime_budget(frozen)
    contract = _architecture_contract(runtime)
    checks = _checks(seed_metrics, runtime)
    if not all(checks.values()):
        raise RuntimeError(f"reduced slow-atlas readiness failed: {checks}")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "local_atlas_seed.npz", seed_arrays)
    _write_json(
        CANONICAL_DIRECTORY / "readiness_metrics.json",
        {"checks": checks, "passed": True, **seed_metrics, "runtime": runtime},
    )
    _write_json(CANONICAL_DIRECTORY / "architecture_contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "field_hashes": frozen["field_hashes"],
            "prior_architecture_hashes": frozen["prior_hashes"],
            "validation_arrays_sha256": _sha(VALIDATION_ARRAYS),
            "field_arrays_sha256": _sha(FIELD_ARRAYS),
            "geometry_arrays_sha256": _sha(GEOMETRY_ARRAYS),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "validated_local_field_preserved": True,
        "selected_online_architecture": "cellwise_Q5_FV_plus_a2_finite_memory_hybrid",
        "local_field_direct_cycle_integration_authorized": False,
        "local_slaving_preflight_authorized": True,
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "online_reduced_solver_implementation_authorized": False,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        field_manifest.THIS_RUNNER,
        field_manifest.THIS_TEST,
        prior_architecture.THIS_RUNNER,
        prior_architecture.THIS_TEST,
    )
    _write_json(
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
            "source_hashes": {relative: _sha(ROOT / relative) for relative in source_files},
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in field_manifest.training._thread_environment()
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
                "# Reduced slow-atlas integrator architecture WP10c9d6c7c3b5c4f25d",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The independently validated 470-coordinate field is retained as a cheap offline atlas layer, not as the cycle-time integrator. Direct microstepping of that field at the previously certified truth scale would still require centuries per 6.7-day cycle.",
                "",
                "The online target remains the conservative 16-cell Q5 finite-volume state plus two stable amplitudes, a stable passive memory kernel of order 0/2/4/6, and a cold/hot/transition branch label. The online system must eliminate the fast stability scale and use multi-second macrosteps.",
                "",
                f"The 13-sample seed atlas has maximum full-state, full-coordinate, q162, and physical-Jacobian errors `{seed_metrics['maximum_seed_full_state_rate_relative_error']:.6e}`, `{seed_metrics['maximum_seed_full_coordinate_rate_relative_error']:.6e}`, `{seed_metrics['maximum_seed_q162_rate_relative_error']:.6e}`, and `{seed_metrics['maximum_seed_q162_Jacobian_relative_error']:.6e}`.",
                "",
                f"At the measured `{runtime['validated_field_median_rhs_wall_seconds']:.6e} s` atlas RHS cost, 100,000 macrosteps with eight atlas evaluations each would spend about `{runtime['reference_100k_macrostep_rhs_wall_seconds']:.1f} s` in RHS work. Runtime is therefore plausible only after the fast timescale is removed; local field speed alone is not sufficient.",
                "",
                "The next package is a no-new-truth local slaving, spectral-gap, conservative-projection, and finite-memory preflight. Failure must expand the conservative macrostate rather than relax stability or error gates.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. No online solver, microburst, exploratory cycle, predictive cycle, or reduced slow evolution is authorized.",
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
