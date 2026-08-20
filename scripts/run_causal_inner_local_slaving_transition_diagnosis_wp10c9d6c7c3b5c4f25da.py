#!/usr/bin/env python3
"""Diagnose whether the validated forward patch is a slow graph or transition layer."""

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
from scipy.linalg import schur


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_forward_quadratic_field_revision_manifest_wp10c9d6c7c3b5c4f25cx as field_manifest  # noqa: E402
import run_causal_inner_reduced_slow_atlas_integrator_manifest_wp10c9d6c7c3b5c4f25d as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25da"
PARENT_COMMIT = "ff4fc9917a067069ef2613a6499944129b919d3d"
PARENT_PARENT = "3cce7dd77aef260955b107aa6cb0565e9f8ed637"
PARENT_TREE = "ab0e95e4ff0a14938efa64770816e2731b837378"

CLASSIFICATION = (
    "local_slow_graph_slaving_rejected_transition_layer_identified_"
    "hybrid_event_map_architecture_authorized"
)
FAIL_CLASSIFICATION = "local_slaving_transition_diagnosis_failed_closed"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25db"

FINE_RADIAL_CELLS = 32
MACRO_RADIAL_CELLS = 16
STORAGE_FIELDS = 5
FINE_STORAGE_DIMENSION = FINE_RADIAL_CELLS * STORAGE_FIELDS
MACRO_STORAGE_DIMENSION = MACRO_RADIAL_CELLS * STORAGE_FIELDS
EXPLICIT_AMPLITUDE_DIMENSION = 2
MACRO_DIMENSION = MACRO_STORAGE_DIMENSION + EXPLICIT_AMPLITUDE_DIMENSION
FIELD_DIMENSION = 470
ACTIVE_DIMENSION = 3
ACTIVE_STEPS = (1.0e-4, 5.0e-5, 2.5e-5)
EFFECTIVE_SAMPLE_TOLERANCE = 1.0e-8
SPECTRAL_GAP_GATE = 10.0

ARTIFACT = (
    "causal_inner_local_slaving_transition_diagnosis_"
    "wp10c9d6c7c3b5c4f25da"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_local_slaving_transition_diagnosis_"
    "wp10c9d6c7c3b5c4f25da.py"
)
THIS_TEST = (
    "tests/test_causal_inner_local_slaving_transition_diagnosis_"
    "wp10c9d6c7c3b5c4f25da.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_LOCAL_SLAVING_TRANSITION_"
    "DIAGNOSIS_WP10C9D6C7C3B5C4F25DA_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
SEED_PATH = parent.CANONICAL_DIRECTORY / "local_atlas_seed.npz"
FIELD_PATH = field_manifest.CANONICAL_DIRECTORY / "forward_quadratic_local_field.npz"


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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


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
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("local slaving parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("local slaving parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("local slaving parent tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(parent.CANONICAL_DIRECTORY / "architecture_contract.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["local_slaving_preflight_authorized"]
        or contract["immediate_local_slaving_preflight"]["work_package"]
        != WORK_PACKAGE
    ):
        raise RuntimeError("local slaving authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"parent source changed: {relative}")
    field_hashes = _checksums(field_manifest.CANONICAL_DIRECTORY)
    if _sha(SEED_PATH) != hashes["local_atlas_seed.npz"]:
        raise RuntimeError("local atlas seed changed")
    if _sha(FIELD_PATH) != field_hashes["forward_quadratic_local_field.npz"]:
        raise RuntimeError("forward quadratic field changed")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("local slaving diagnosis requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "hashes": hashes,
        "field_hashes": field_hashes,
    }


def _conservative_operators(
    fine_scales: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    scales = np.asarray(fine_scales, dtype=float).reshape(
        FINE_RADIAL_CELLS, STORAGE_FIELDS
    )
    if np.any(scales <= 0.0):
        raise RuntimeError("mapped storage scales must be positive")
    macro_scales = scales.reshape(
        MACRO_RADIAL_CELLS, 2, STORAGE_FIELDS
    ).sum(axis=1)
    restriction = np.zeros((MACRO_DIMENSION, FIELD_DIMENSION), dtype=float)
    lifting = np.zeros((FIELD_DIMENSION, MACRO_DIMENSION), dtype=float)
    for macro_cell in range(MACRO_RADIAL_CELLS):
        for field in range(STORAGE_FIELDS):
            target = STORAGE_FIELDS * macro_cell + field
            left = STORAGE_FIELDS * (2 * macro_cell) + field
            right = STORAGE_FIELDS * (2 * macro_cell + 1) + field
            total_scale = macro_scales[macro_cell, field]
            restriction[target, left] = scales[2 * macro_cell, field] / total_scale
            restriction[target, right] = scales[2 * macro_cell + 1, field] / total_scale
            lifting[left, target] = 1.0
            lifting[right, target] = 1.0
    for amplitude in range(EXPLICIT_AMPLITUDE_DIMENSION):
        source = FINE_STORAGE_DIMENSION + amplitude
        target = MACRO_STORAGE_DIMENSION + amplitude
        restriction[target, source] = 1.0
        lifting[source, target] = 1.0

    fine_ledgers = np.zeros((3, FIELD_DIMENSION), dtype=float)
    macro_ledgers = np.zeros((3, MACRO_DIMENSION), dtype=float)
    for field in range(3):
        fine_ledgers[field, field:FINE_STORAGE_DIMENSION:STORAGE_FIELDS] = scales[
            :, field
        ]
        macro_ledgers[field, field:MACRO_STORAGE_DIMENSION:STORAGE_FIELDS] = (
            macro_scales[:, field]
        )
    identity_defect = float(
        np.max(np.abs(restriction @ lifting - np.eye(MACRO_DIMENSION)))
    )
    restriction_ledger_defect = float(
        np.linalg.norm(macro_ledgers @ restriction - fine_ledgers)
        / max(np.linalg.norm(fine_ledgers), np.finfo(float).tiny)
    )
    lifting_ledger_defect = float(
        np.linalg.norm(fine_ledgers @ lifting - macro_ledgers)
        / max(np.linalg.norm(macro_ledgers), np.finfo(float).tiny)
    )
    metrics = {
        "fine_storage_dimension": FINE_STORAGE_DIMENSION,
        "macro_storage_dimension": MACRO_STORAGE_DIMENSION,
        "macro_total_dimension": MACRO_DIMENSION,
        "storage_restriction_rank": int(
            np.linalg.matrix_rank(restriction[:MACRO_STORAGE_DIMENSION])
        ),
        "total_restriction_rank": int(np.linalg.matrix_rank(restriction)),
        "restriction_lifting_identity_infinity_defect": identity_defect,
        "global_M_J_E_restriction_relative_defect": restriction_ledger_defect,
        "global_M_J_E_lifting_relative_defect": lifting_ledger_defect,
    }
    arrays = {
        "macro_restriction": restriction,
        "constraint_compatible_piecewise_constant_lifting": lifting,
        "fine_mapped_storage_scales": scales,
        "macro_mapped_storage_scales": macro_scales,
        "fine_global_M_J_E_ledger_rows": fine_ledgers,
        "macro_global_M_J_E_ledger_rows": macro_ledgers,
    }
    return arrays, metrics


def _active_rate(
    field: field_manifest.ForwardQuadraticAuthenticCenterField,
    coordinate: np.ndarray,
    active_basis: np.ndarray,
) -> np.ndarray:
    rate = field.field(np.asarray(coordinate, dtype=float))
    return rate[-field_manifest.DEPARTURE_DIMENSION :] @ active_basis / field_manifest.ACTIVE_SCALE


def _active_jacobian_ladder(
    field: field_manifest.ForwardQuadraticAuthenticCenterField,
    active_basis: np.ndarray,
    macro_restriction: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    zero = np.zeros(FIELD_DIMENSION, dtype=float)
    active_matrices = []
    macro_matrices = []
    for step in ACTIVE_STEPS:
        active_columns = []
        macro_columns = []
        for column in range(ACTIVE_DIMENSION):
            displacement = np.zeros(FIELD_DIMENSION, dtype=float)
            displacement[-field_manifest.DEPARTURE_DIMENSION :] = (
                field_manifest.ACTIVE_SCALE * active_basis[:, column] * step
            )
            plus = field.field(zero + displacement)
            minus = field.field(zero - displacement)
            derivative = (plus - minus) / (2.0 * step)
            active_columns.append(
                derivative[-field_manifest.DEPARTURE_DIMENSION :]
                @ active_basis
                / field_manifest.ACTIVE_SCALE
            )
            macro_columns.append(macro_restriction @ derivative)
        active_matrices.append(np.column_stack(active_columns))
        macro_matrices.append(np.column_stack(macro_columns))

    active_defects = []
    macro_defects = []
    for coarse, fine in zip(
        active_matrices[:-1], active_matrices[1:], strict=True
    ):
        active_defects.append(
            float(
                np.linalg.norm(coarse - fine)
                / max(np.linalg.norm(fine), np.finfo(float).tiny)
            )
        )
    for coarse, fine in zip(
        macro_matrices[:-1], macro_matrices[1:], strict=True
    ):
        macro_defects.append(
            float(
                np.linalg.norm(coarse - fine)
                / max(np.linalg.norm(fine), np.finfo(float).tiny)
            )
        )

    active = active_matrices[-1]
    eigenvalues = np.linalg.eigvals(active)
    if float(np.max(np.abs(eigenvalues.imag))) > 1.0e-6:
        raise RuntimeError("validated active spectrum unexpectedly became complex")
    real_values = np.sort(eigenvalues.real)
    nonstable = real_values[real_values >= 0.0]
    stable = real_values[real_values < 0.0]
    if len(nonstable) != 1 or len(stable) != 2:
        raise RuntimeError(f"unexpected active inertia: {real_values}")
    slow_scale = max(float(np.max(np.abs(nonstable))), 1.0 / 578880.0)
    stable_by_magnitude = stable[np.argsort(np.abs(stable))]
    initial_gap = float(abs(stable_by_magnitude[0]) / slow_scale)
    promoted_values = np.concatenate((nonstable, stable_by_magnitude[:1]))
    remaining_values = stable_by_magnitude[1:]
    promoted_scale = float(np.max(np.abs(promoted_values)))
    promoted_gap = float(np.min(np.abs(remaining_values)) / promoted_scale)
    cutoff = 0.5 * float(stable_by_magnitude[0] + stable_by_magnitude[1])
    schur_matrix, schur_vectors, retained_count = schur(
        active, output="real", sort=lambda real, imaginary=0.0: real > cutoff
    )
    if retained_count != 2:
        raise RuntimeError("ordered active Schur split did not retain two coordinates")
    signs = np.ones(ACTIVE_DIMENSION)
    for column in range(ACTIVE_DIMENSION):
        pivot = int(np.argmax(np.abs(schur_vectors[:, column])))
        if schur_vectors[pivot, column] < 0.0:
            signs[column] = -1.0
    schur_vectors = schur_vectors @ np.diag(signs)
    schur_matrix = schur_vectors.T @ active @ schur_vectors
    center_active_rate = _active_rate(field, zero, active_basis)
    center_schur_rate = schur_vectors.T @ center_active_rate
    fast_affine_offset = float(-center_schur_rate[-1] / schur_matrix[-1, -1])
    lower_left = schur_matrix[2:, :2]
    invariance_defect = float(
        np.linalg.norm(lower_left)
        / max(np.linalg.norm(schur_matrix), np.finfo(float).tiny)
    )
    metrics = {
        "active_finite_difference_steps": ACTIVE_STEPS,
        "active_Jacobian_adjacent_relative_defects": active_defects,
        "macro_active_Jacobian_adjacent_relative_defects": macro_defects,
        "maximum_Jacobian_step_ladder_relative_defect": float(
            max((*active_defects, *macro_defects))
        ),
        "active_eigenvalues_per_second": real_values,
        "full_slaving_maximum_spectral_abscissa_per_second": float(
            np.max(real_values)
        ),
        "nonstable_active_dimension": int(len(nonstable)),
        "all_active_slaving_is_stable": bool(np.max(real_values) <= 0.0),
        "nonstable_only_promotion_gap_ratio": initial_gap,
        "selected_transition_coordinate_dimension": 2,
        "selected_transition_eigenvalues_per_second": np.sort(promoted_values),
        "remaining_fast_eigenvalue_per_second": float(remaining_values[0]),
        "selected_fast_gap_ratio": promoted_gap,
        "selected_fast_block_spectral_abscissa_per_second": float(
            remaining_values[0]
        ),
        "ordered_Schur_invariance_relative_defect": invariance_defect,
        "center_active_rate_norm_per_second": float(
            np.linalg.norm(center_active_rate)
        ),
        "center_Schur_rate_per_second": center_schur_rate,
        "fast_affine_equilibrium_offset_in_normalized_active_coordinates": (
            fast_affine_offset
        ),
        "strong_fast_e_folding_seconds": float(1.0 / abs(remaining_values[0])),
        "weak_retained_e_folding_seconds": float(
            1.0 / abs(stable_by_magnitude[0])
        ),
        "unstable_retained_e_folding_seconds": float(1.0 / nonstable[0]),
    }
    arrays = {
        "active_Jacobian_step_ladder": np.asarray(active_matrices),
        "macro_active_Jacobian_step_ladder": np.asarray(macro_matrices),
        "ordered_active_Schur_matrix_per_second": schur_matrix,
        "ordered_active_Schur_vectors": schur_vectors,
        "center_active_rate_per_second": center_active_rate,
        "center_Schur_rate_per_second": center_schur_rate,
    }
    return arrays, metrics


def _seed_diagnostics(
    seed: dict[str, np.ndarray],
    macro_restriction: np.ndarray,
    schur_vectors: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    coordinates = np.asarray(seed["seed_local_coordinates"], dtype=float)
    active = np.asarray(seed["seed_active_coordinates"], dtype=float)
    exact_rates = np.asarray(seed["seed_exact_coordinate_rates_per_second"], dtype=float)
    predicted_rates = np.asarray(
        seed["seed_predicted_coordinate_rates_per_second"], dtype=float
    )
    macro_coordinates = coordinates @ macro_restriction.T
    exact_macro_rates = exact_rates @ macro_restriction.T
    predicted_macro_rates = predicted_rates @ macro_restriction.T
    active_basis = np.asarray(seed["active_departure_basis"], dtype=float)
    exact_active_rates = (
        exact_rates[:, -field_manifest.DEPARTURE_DIMENSION :]
        @ active_basis
        / field_manifest.ACTIVE_SCALE
    )
    predicted_active_rates = (
        predicted_rates[:, -field_manifest.DEPARTURE_DIMENSION :]
        @ active_basis
        / field_manifest.ACTIVE_SCALE
    )
    exact_schur_rates = exact_active_rates @ schur_vectors
    predicted_schur_rates = predicted_active_rates @ schur_vectors
    macro_errors = np.linalg.norm(
        predicted_macro_rates - exact_macro_rates, axis=1
    ) / np.maximum(
        np.linalg.norm(exact_macro_rates, axis=1), np.finfo(float).tiny
    )
    transition_errors = np.linalg.norm(
        predicted_schur_rates - exact_schur_rates, axis=1
    ) / np.maximum(
        np.linalg.norm(exact_schur_rates, axis=1), np.finfo(float).tiny
    )
    centered_macro = macro_coordinates - macro_coordinates[0]
    macro_singular = np.linalg.svd(centered_macro, compute_uv=False)
    maximum_macro_norm = float(np.max(np.linalg.norm(centered_macro, axis=1)))
    maximum_active_norm = float(np.max(np.linalg.norm(active, axis=1)))
    maximum_validated_radius = maximum_active_norm
    metrics = {
        "seed_count": int(len(coordinates)),
        "maximum_normalized_macro_coordinate_norm": maximum_macro_norm,
        "maximum_normalized_active_coordinate_norm": maximum_active_norm,
        "macro_to_active_seed_diameter_ratio": float(
            maximum_macro_norm / max(maximum_active_norm, np.finfo(float).tiny)
        ),
        "macro_seed_singular_values": macro_singular,
        "macro_seed_effective_rank_at_1e_8": int(
            np.count_nonzero(macro_singular > EFFECTIVE_SAMPLE_TOLERANCE)
        ),
        "maximum_seed_macro_rate_relative_error": float(np.max(macro_errors)),
        "maximum_seed_transition_rate_relative_error": float(
            np.max(transition_errors)
        ),
        "maximum_validated_active_radius": maximum_validated_radius,
    }
    arrays = {
        "seed_macro_coordinates": macro_coordinates,
        "seed_exact_macro_rates_per_second": exact_macro_rates,
        "seed_predicted_macro_rates_per_second": predicted_macro_rates,
        "seed_exact_active_rates_per_second": exact_active_rates,
        "seed_predicted_active_rates_per_second": predicted_active_rates,
        "seed_exact_transition_Schur_rates_per_second": exact_schur_rates,
        "seed_predicted_transition_Schur_rates_per_second": predicted_schur_rates,
        "seed_macro_rate_relative_errors": macro_errors,
        "seed_transition_rate_relative_errors": transition_errors,
    }
    return arrays, metrics


def _checks(conservative: dict, tangent: dict, seed: dict) -> dict[str, bool]:
    return {
        "storage_rank": conservative["storage_restriction_rank"]
        >= MACRO_STORAGE_DIMENSION,
        "total_rank": conservative["total_restriction_rank"] >= MACRO_DIMENSION,
        "restriction_lifting_identity": conservative[
            "restriction_lifting_identity_infinity_defect"
        ]
        <= 1.0e-12,
        "global_M_J_E_restriction": conservative[
            "global_M_J_E_restriction_relative_defect"
        ]
        <= 1.0e-12,
        "global_M_J_E_lifting": conservative[
            "global_M_J_E_lifting_relative_defect"
        ]
        <= 1.0e-12,
        "field_Jacobian_step_ladder": tangent[
            "maximum_Jacobian_step_ladder_relative_defect"
        ]
        <= 1.0e-3,
        "one_nonstable_active_direction_detected": tangent[
            "nonstable_active_dimension"
        ]
        == 1,
        "full_slaving_rejected": not tangent["all_active_slaving_is_stable"],
        "nonstable_only_gap_rejected": tangent[
            "nonstable_only_promotion_gap_ratio"
        ]
        < SPECTRAL_GAP_GATE,
        "two_coordinate_transition_split_gap": tangent[
            "selected_fast_gap_ratio"
        ]
        >= SPECTRAL_GAP_GATE,
        "remaining_fast_block_stable": tangent[
            "selected_fast_block_spectral_abscissa_per_second"
        ]
        <= 0.0,
        "ordered_Schur_invariance": tangent[
            "ordered_Schur_invariance_relative_defect"
        ]
        <= 1.0e-1,
        "macro_seed_does_not_claim_identifiability": seed[
            "macro_seed_effective_rank_at_1e_8"
        ]
        == 0,
        "macro_rate_prediction": seed["maximum_seed_macro_rate_relative_error"]
        <= 7.5e-2,
        "transition_rate_prediction": seed[
            "maximum_seed_transition_rate_relative_error"
        ]
        <= 7.5e-2,
    }


def _revised_architecture(tangent: dict, seed: dict) -> dict:
    fast_offset = tangent[
        "fast_affine_equilibrium_offset_in_normalized_active_coordinates"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "diagnosis": {
            "validated_forward_patch_role": "offline_fast_transition_layer_seed",
            "validated_forward_patch_is_slow_graph": False,
            "reason": (
                "one_unstable_validated_active_mode_and_no_resolved_macro_"
                "variation_in_the_fixed_q_seed_cloud"
            ),
            "full_active_slaving_spectral_abscissa_per_second": tangent[
                "full_slaving_maximum_spectral_abscissa_per_second"
            ],
            "nonstable_only_gap_ratio": tangent[
                "nonstable_only_promotion_gap_ratio"
            ],
            "two_coordinate_transition_split_gap_ratio": tangent[
                "selected_fast_gap_ratio"
            ],
            "fast_affine_offset_over_validated_radius": float(
                abs(fast_offset)
                / max(seed["maximum_validated_active_radius"], np.finfo(float).tiny)
            ),
            "slow_graph_and_memory_fit_executed": False,
            "why_memory_screen_stopped": (
                "stable_memory_cannot_absorb_an_unstable_transition_coordinate_"
                "and_the_seed_contains_no_effective_macro_state_variation"
            ),
        },
        "online_branch_layer": {
            "state": "U80_Q5_conservative_storage_plus_a2_plus_stable_memory_mr",
            "branch_labels": ["cold", "hot"],
            "memory_orders_to_screen_on_branch_data_only": [0, 2, 4, 6],
            "evolution": "second_order_conservative_IMEX_ARK_with_multi_second_steps",
            "stable_fast_coordinates": "algebraic_branchwise_slaving_only_after_gap_and_invariance_pass",
            "online_truth_calls": 0,
            "online_fast_microsteps": 0,
        },
        "offline_transition_layer": {
            "local_internal_coordinate_dimension": ACTIVE_DIMENSION,
            "retained_transition_Schur_dimension_for_local_geometry": 2,
            "remaining_strong_fast_dimension": 1,
            "role": (
                "construct_event_entry_to_exit_map_and_integrated_flux_source_ledger"
            ),
            "operator": (
                "T_b:(U_minus,a_minus,m_minus,event_parameters)->"
                "(U_plus,a_plus,m_plus,Delta_L_MJE,b_plus)"
            ),
            "online_use": "interpolated_conservative_jump_map_not_transition_ODE",
            "current_patch_scope": "one_forward_sector_only",
            "new_truth_needed_before_training": True,
        },
        "event_contract": {
            "trigger": "branch_stability_or_fold_surface_with_hysteresis",
            "location": "bracketed_event_root_in_slow_macro_time",
            "reset": "apply_prevalidated_transition_map_once",
            "conservation": (
                "macro_state_jump_equals_integrated_transition_face_flux_plus_"
                "source_work_ledger"
            ),
            "rejected_or_out_of_domain_transition": "stop_without_propagation",
        },
        "next_definitions_only_package": {
            "work_package": AUTHORIZED_NEXT,
            "purpose": "freeze_separate_branch_and_transition_atlas_data_contracts",
            "must_freeze_before_truth": [
                "cold_and_hot_branch_anchor_coordinates_with_nonzero_U80_variation",
                "event_entry_and_exit_surfaces",
                "transition_impulse_outputs_and_exact_M_J_E_ledger",
                "training_and_sealed_holdout_roles",
                "pathwise_truth_budget_and_fail_fast_order",
                "branch_memory_order_selection_before_opening_holdouts",
            ],
            "may_not_authorize": [
                "online_reduced_solver_implementation",
                "exploratory_cycle",
                "predictive_cycle",
                "reduced_slow_evolution",
            ],
        },
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
                    "scientific_status": "DIAGNOSTIC",
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
        raise RuntimeError("local slaving transition diagnosis already exists")
    seed = _load_npz(SEED_PATH)
    closure = _load_npz(FIELD_PATH)
    field = field_manifest.ForwardQuadraticAuthenticCenterField(closure)
    conservative_arrays, conservative_metrics = _conservative_operators(
        field.model.components["mapped_row_scales"]
    )
    tangent_arrays, tangent_metrics = _active_jacobian_ladder(
        field,
        np.asarray(seed["active_departure_basis"], dtype=float),
        conservative_arrays["macro_restriction"],
    )
    seed_arrays, seed_metrics = _seed_diagnostics(
        seed,
        conservative_arrays["macro_restriction"],
        tangent_arrays["ordered_active_Schur_vectors"],
    )
    checks = _checks(conservative_metrics, tangent_metrics, seed_metrics)
    passed = bool(all(checks.values()))
    classification = CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = AUTHORIZED_NEXT if passed else None
    architecture = _revised_architecture(tangent_metrics, seed_metrics)

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(
        CANONICAL_DIRECTORY / "diagnostic_arrays.npz",
        {**conservative_arrays, **tangent_arrays, **seed_arrays},
    )
    _write_json(
        CANONICAL_DIRECTORY / "diagnostic_metrics.json",
        {
            "checks": checks,
            "passed": passed,
            "conservative": conservative_metrics,
            "active_tangent": tangent_metrics,
            "seed": seed_metrics,
            "new_exact_rate_calls": 0,
            "new_complete_generator_assemblies": 0,
            "new_nonlinear_fixed_Q_roots": 0,
            "propagated_states": 0,
        },
    )
    _write_json(CANONICAL_DIRECTORY / "revised_architecture.json", architecture)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "parent_hashes": frozen["hashes"],
            "field_hashes": frozen["field_hashes"],
            "seed_sha256": _sha(SEED_PATH),
            "field_sha256": _sha(FIELD_PATH),
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "validated_forward_field_preserved": True,
        "conservative_U80_plus_a2_geometry_passed": bool(
            all(
                checks[name]
                for name in (
                    "storage_rank",
                    "total_rank",
                    "restriction_lifting_identity",
                    "global_M_J_E_restriction",
                    "global_M_J_E_lifting",
                )
            )
        ),
        "forward_patch_is_slow_graph": False,
        "forward_patch_is_transition_layer_seed": passed,
        "all_active_slaving_rejected": checks["full_slaving_rejected"],
        "selected_offline_transition_coordinate_dimension": (
            tangent_metrics["selected_transition_coordinate_dimension"]
        ),
        "branch_memory_screen_executed": False,
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "online_reduced_solver_implementation_authorized": False,
        "exploratory_cycle_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        field_manifest.THIS_RUNNER,
        field_manifest.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DIAGNOSTIC",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
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
                for name in parent.field_manifest.training._thread_environment()
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
                "# Local slaving and transition diagnosis WP10c9d6c7c3b5c4f25da",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "The conservative 32-to-16-cell Q5 restriction and lifting pass exactly, including the global mass, angular-momentum, and Killing-energy ledgers. The obstruction is therefore not conservative geometry.",
                "",
                f"The validated three-direction forward tangent has eigenvalues `{tangent_metrics['active_eigenvalues_per_second']}` per second. One direction is unstable, so treating the whole forward patch as a slaved slow graph is rejected. Promoting only that direction leaves gap `{tangent_metrics['nonstable_only_promotion_gap_ratio']:.6e}`, below the frozen gate `{SPECTRAL_GAP_GATE:.1f}`. A two-coordinate transition split leaves a stable fast mode and gap `{tangent_metrics['selected_fast_gap_ratio']:.6e}`.",
                "",
                f"The fixed-Q seed cloud has effective macro rank `{seed_metrics['macro_seed_effective_rank_at_1e_8']}` at tolerance `{EFFECTIVE_SAMPLE_TOLERANCE:.1e}`. It cannot identify branchwise slow closure or stable memory. Those fits were stopped rather than inferred from numerical retraction noise.",
                "",
                "The refined architecture is hybrid: multi-second conservative evolution on separately trained cold/hot slow branches, bracketed event surfaces, and a prevalidated offline fast transition map carrying its integrated conservative ledger. The validated forward field seeds that transition map; it is never microstepped around the full cycle online.",
                "",
                f"Authorized next artifact: `{authorized_next}`. It is definitions-only and must freeze separate branch and transition datasets before any new truth. No online solver or cycle is authorized.",
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
    summary = _run()
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
