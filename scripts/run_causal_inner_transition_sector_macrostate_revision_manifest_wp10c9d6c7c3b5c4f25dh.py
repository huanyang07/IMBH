#!/usr/bin/env python3
"""Reconcile the exact hidden anchor rate with the hybrid transition architecture."""

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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_hybrid_candidate_geometry_preflight_wp10c9d6c7c3b5c4f25dc as candidate  # noqa: E402
import run_causal_inner_local_slaving_transition_diagnosis_wp10c9d6c7c3b5c4f25da as transition  # noqa: E402
import run_causal_inner_primary_hidden_anchor_preflight_wp10c9d6c7c3b5c4f25dg as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dh"
PARENT_COMMIT = "ac0d86a548fef2f7305dd8246dbba36ecf0bdbac"
PARENT_PARENT = "b6464fbb1931dda0550d0065a828b26bd18cae89"
PARENT_TREE = "d3b85d0fd545cfa1ee2da2a9cb58fc988dd3450e"

CLASSIFICATION = (
    "exact_primary_transition_sector_reconciled_"
    "three_coordinate_internal_model_rejected_"
    "rank_adaptive_hidden_impulse_map_architecture_frozen"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25di"

COORDINATE_DIMENSION = 470
MACRO_DIMENSION = 82
HIDDEN_DIMENSION = 388
FINE_STORAGE_DIMENSION = 160
EXPLICIT_AMPLITUDE_DIMENSION = 2
MEMORY_DIMENSION = 280
DEPARTURE_DIMENSION = 28
GAUGE_DIMENSION = 90
PHYSICAL_DIMENSION = 560
RADIAL_CELLS = 112
PHYSICAL_FIELDS = 5

GEOMETRY_GATE = 5.0e-12
ENERGY_PARTITION_GATE = 5.0e-12
MEMORY_DEPARTURE_ENERGY_MIN = 0.99
PRIOR_TWO_MODE_TOTAL_CAPTURE_MAX = 0.25
PRIOR_THREE_MODE_TOTAL_CAPTURE_MAX = 0.50

ARTIFACT = (
    "causal_inner_transition_sector_macrostate_revision_manifest_"
    "wp10c9d6c7c3b5c4f25dh"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_transition_sector_macrostate_revision_manifest_"
    "wp10c9d6c7c3b5c4f25dh.py"
)
THIS_TEST = (
    "tests/test_causal_inner_transition_sector_macrostate_revision_manifest_"
    "wp10c9d6c7c3b5c4f25dh.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRANSITION_SECTOR_MACROSTATE_"
    "REVISION_MANIFEST_WP10C9D6C7C3B5C4F25DH_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_RATE_ARRAYS = parent.CANONICAL_DIRECTORY / "primary_anchor_rate_arrays.npz"
PARENT_RATE_METRICS = parent.CANONICAL_DIRECTORY / "primary_anchor_rate_metrics.json"
TRANSITION_ARRAYS = transition.CANONICAL_DIRECTORY / "diagnostic_arrays.npz"
TRANSITION_METRICS = transition.CANONICAL_DIRECTORY / "diagnostic_metrics.json"
TRANSITION_ARCHITECTURE = transition.CANONICAL_DIRECTORY / "revised_architecture.json"
CANDIDATE_ARRAYS = candidate.CANONICAL_DIRECTORY / "candidate_geometry_arrays.npz"
EXACT_CHART_ARRAYS = parent.manifest.CHART_DIRECTORY / "exact_chart_arrays.npz"
LOCAL_ATLAS_SEED = transition.SEED_PATH


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
        raise RuntimeError("transition-revision parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("transition-revision parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("transition-revision parent tree changed")

    parent_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    parent_summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    parent_metrics = _read(PARENT_RATE_METRICS)
    parent_provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    root_contract = _read(
        parent.manifest.CANONICAL_DIRECTORY / "primary_hidden_root_contract.json"
    )
    decision = root_contract["decision"]["anchor_hidden_fraction_fail"]
    if (
        parent_summary["passed"]
        or parent_summary["classification"] != decision["classification"]
        or parent_summary["anchor_hidden_fraction_gate_passed"]
        or parent_summary["hidden_root_attempted"]
        or parent_summary["complete_generator_assembled"]
        or parent_summary["new_exact_fixed_Q_rate_evaluations"] != 1
        or parent_summary["new_complete_generator_assemblies"] != 0
        or parent_summary["new_intrinsic_hidden_roots"] != 0
        or parent_summary["sealed_16ms_opened"]
        or "transition_or_macro_state_revision" not in decision["authorizes_only"]
        or not all(parent_metrics["checks"].values())
        or parent_metrics["hidden_fraction_gate_passed"]
    ):
        raise RuntimeError("hidden-anchor rejection contract changed")
    for relative, expected in parent_provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"hidden-anchor source changed: {relative}")

    transition_hashes = _checksums(transition.CANONICAL_DIRECTORY)
    transition_summary = _read(transition.CANONICAL_DIRECTORY / "summary.json")
    transition_metrics = _read(TRANSITION_METRICS)
    transition_architecture = _read(TRANSITION_ARCHITECTURE)
    if (
        not transition_summary["passed"]
        or not transition_summary["all_active_slaving_rejected"]
        or not transition_summary["forward_patch_is_transition_layer_seed"]
        or transition_summary["selected_offline_transition_coordinate_dimension"]
        != 2
        or not all(transition_metrics["checks"].values())
        or transition_architecture["online_branch_layer"]["online_fast_microsteps"]
        != 0
    ):
        raise RuntimeError("prior transition-layer diagnosis changed")

    candidate_hashes = _checksums(candidate.CANONICAL_DIRECTORY)
    candidate_summary = _read(candidate.CANONICAL_DIRECTORY / "summary.json")
    candidate_arrays = _load_npz(CANDIDATE_ARRAYS)
    primary_index = int(np.asarray(candidate_arrays["primary_candidate_index"]))
    sealed_index = int(np.asarray(candidate_arrays["sealed_candidate_index"]))
    if (
        not candidate_summary["passed"]
        or candidate_summary["primary_candidate"] != "U20_unclassified_primary"
        or candidate_summary["sealed_candidate"] != "U16_unclassified_sealed"
        or not candidate_summary["all_candidates_unclassified"]
        or candidate_summary["new_exact_rate_calls"] != 0
        or float(candidate_arrays["candidate_times_seconds"][primary_index])
        != candidate.PRIMARY_TIME_SECONDS
        or float(candidate_arrays["candidate_times_seconds"][sealed_index])
        != candidate.SEALED_TIME_SECONDS
    ):
        raise RuntimeError("hybrid candidate geometry changed")

    for name, expected in parent_provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("transition-sector revision requires a clean tracked tree")
    return {
        "parent_hashes": parent_hashes,
        "transition_hashes": transition_hashes,
        "candidate_hashes": candidate_hashes,
        "parent_classification": parent_summary["classification"],
        "transition_classification": transition_summary["classification"],
    }


def _block_slices() -> dict[str, slice]:
    return {
        "unresolved_storage80": slice(0, FINE_STORAGE_DIMENSION),
        "explicit_amplitude2": slice(
            FINE_STORAGE_DIMENSION,
            FINE_STORAGE_DIMENSION + EXPLICIT_AMPLITUDE_DIMENSION,
        ),
        "memory280": slice(
            FINE_STORAGE_DIMENSION + EXPLICIT_AMPLITUDE_DIMENSION,
            FINE_STORAGE_DIMENSION + EXPLICIT_AMPLITUDE_DIMENSION + MEMORY_DIMENSION,
        ),
        "departure28": slice(COORDINATE_DIMENSION - DEPARTURE_DIMENSION, None),
    }


def _projection_capture(
    hidden_action: np.ndarray,
    active_basis: np.ndarray,
    schur_vectors: np.ndarray,
    count: int,
) -> tuple[np.ndarray, float, float]:
    departure_basis = active_basis @ schur_vectors[:, :count]
    projected = np.zeros(COORDINATE_DIMENSION, dtype=float)
    departure = np.asarray(hidden_action[-DEPARTURE_DIMENSION:], dtype=float)
    projected[-DEPARTURE_DIMENSION:] = departure_basis @ (
        departure_basis.T @ departure
    )
    total_denominator = max(
        float(np.linalg.norm(hidden_action) ** 2), np.finfo(float).tiny
    )
    departure_denominator = max(
        float(np.linalg.norm(departure) ** 2), np.finfo(float).tiny
    )
    energy = float(np.linalg.norm(projected) ** 2)
    return projected, energy / total_denominator, energy / departure_denominator


def _reconciliation() -> tuple[dict, dict[str, np.ndarray]]:
    rate = _load_npz(PARENT_RATE_ARRAYS)
    parent_metrics = _read(PARENT_RATE_METRICS)["metrics"]
    dual = _load_npz(
        parent.manifest.CANONICAL_DIRECTORY / "dual_hidden_geometry.npz"
    )
    old = _load_npz(TRANSITION_ARRAYS)
    old_payload = _read(TRANSITION_METRICS)
    old_metrics = old_payload["active_tangent"]
    seed = _load_npz(LOCAL_ATLAS_SEED)
    candidates = _load_npz(CANDIDATE_ARRAYS)
    chart = _load_npz(EXACT_CHART_ARRAYS)

    R = np.asarray(dual["macro_restriction_R82"], dtype=float)
    L = np.asarray(dual["macro_lifting_L82"], dtype=float)
    old_R = np.asarray(old["macro_restriction"], dtype=float)
    old_L = np.asarray(
        old["constraint_compatible_piecewise_constant_lifting"], dtype=float
    )
    F = np.asarray(rate["coordinate_rate_F470_per_s"], dtype=float)
    hidden_action = np.asarray(rate["hidden_action_ZH470_per_s"], dtype=float)
    macro_action = np.asarray(rate["macro_action_LG470_per_s"], dtype=float)
    active_basis = np.asarray(seed["active_departure_basis"], dtype=float)
    schur_vectors = np.asarray(old["ordered_active_Schur_vectors"], dtype=float)
    primary_index = int(np.asarray(candidates["primary_candidate_index"]))
    sealed_index = int(np.asarray(candidates["sealed_candidate_index"]))
    anchor_state = np.asarray(rate["anchor_primitive_state"], dtype=float)
    candidate_state = np.asarray(
        candidates["candidate_primitive_states"][primary_index], dtype=float
    )
    anchor_active = np.asarray(
        candidates["candidate_active_coordinates"][primary_index], dtype=float
    )
    validated_radius = float(old_payload["seed"]["maximum_validated_active_radius"])

    hidden_norm_squared = max(
        float(np.linalg.norm(hidden_action) ** 2), np.finfo(float).tiny
    )
    block_energy_fractions = {}
    block_norms = {}
    for name, selection in _block_slices().items():
        block = hidden_action[selection]
        block_norms[name] = float(np.linalg.norm(block))
        block_energy_fractions[name] = float(
            np.linalg.norm(block) ** 2 / hidden_norm_squared
        )

    projected = {}
    total_capture = {}
    departure_capture = {}
    for count in (1, 2, 3):
        action, total, departure = _projection_capture(
            hidden_action, active_basis, schur_vectors, count
        )
        projected[count] = action
        total_capture[count] = total
        departure_capture[count] = departure

    augmented = np.asarray(
        chart["anchor_augmented_chart_jacobian"], dtype=float
    )
    right = np.concatenate((hidden_action, np.zeros(GAUGE_DIMENSION)))
    physical_hidden = np.linalg.solve(augmented, right)
    physical_residual = augmented @ physical_hidden - right
    physical_cells = physical_hidden.reshape(RADIAL_CELLS, PHYSICAL_FIELDS)
    physical_energy = max(
        float(np.linalg.norm(physical_cells) ** 2), np.finfo(float).tiny
    )
    field_energy = np.sum(physical_cells**2, axis=0) / physical_energy
    radial_groups = np.array_split(np.arange(RADIAL_CELLS), 4)
    radial_energy = np.asarray(
        [
            np.linalg.norm(physical_cells[group]) ** 2 / physical_energy
            for group in radial_groups
        ]
    )

    hidden_fraction_recomputed = float(
        np.linalg.norm(hidden_action)
        / max(np.linalg.norm(F), np.finfo(float).tiny)
    )
    metrics = {
        "macro_restriction_bitwise_matches_prior_transition_architecture": bool(
            np.array_equal(R, old_R)
        ),
        "macro_lifting_bitwise_matches_prior_transition_architecture": bool(
            np.array_equal(L, old_L)
        ),
        "anchor_state_bitwise_matches_primary_candidate": bool(
            np.array_equal(anchor_state, candidate_state)
        ),
        "primary_candidate_index": primary_index,
        "sealed_candidate_index_hash_only": sealed_index,
        "hidden_fraction_recomputed": hidden_fraction_recomputed,
        "hidden_fraction_recorded": float(
            parent_metrics["hidden_coordinate_rate_fraction"]
        ),
        "coordinate_rate_decomposition_relative_defect": float(
            np.linalg.norm(macro_action + hidden_action - F)
            / max(np.linalg.norm(F), np.finfo(float).tiny)
        ),
        "hidden_block_norms_per_second": block_norms,
        "hidden_block_energy_fractions": block_energy_fractions,
        "hidden_block_energy_sum_defect": float(
            abs(sum(block_energy_fractions.values()) - 1.0)
        ),
        "memory_plus_departure_hidden_energy_fraction": float(
            block_energy_fractions["memory280"]
            + block_energy_fractions["departure28"]
        ),
        "prior_transition_total_energy_capture": {
            str(key): value for key, value in total_capture.items()
        },
        "prior_transition_departure_energy_capture": {
            str(key): value for key, value in departure_capture.items()
        },
        "residual_hidden_norm_fraction_after_prior_three_modes": float(
            np.linalg.norm(hidden_action - projected[3])
            / max(np.linalg.norm(hidden_action), np.finfo(float).tiny)
        ),
        "anchor_active_coordinates": anchor_active,
        "anchor_active_coordinate_norm": float(np.linalg.norm(anchor_active)),
        "prior_validated_active_radius": validated_radius,
        "anchor_active_radius_over_prior_validated_radius": float(
            np.linalg.norm(anchor_active) / validated_radius
        ),
        "gauge_fixed_physical_hidden_lift_relative_residual": float(
            np.linalg.norm(physical_residual)
            / max(np.linalg.norm(right), np.finfo(float).tiny)
        ),
        "gauge_fixed_physical_hidden_field_energy_fractions": field_energy,
        "gauge_fixed_physical_hidden_radial_quartile_energy_fractions": radial_energy,
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "new_chart_retractions": 0,
        "propagated_states": 0,
        "sealed_16ms_truth_calls": 0,
    }
    arrays = {
        "macro_restriction_R82": R,
        "macro_lifting_L82": L,
        "exact_anchor_coordinate_rate_F470_per_s": F,
        "exact_anchor_hidden_action470_per_s": hidden_action,
        "exact_anchor_macro_action470_per_s": macro_action,
        "prior_active_departure_basis28x3": active_basis,
        "prior_ordered_active_Schur_vectors3x3": schur_vectors,
        "prior_transition1_projected_action470_per_s": projected[1],
        "prior_transition2_projected_action470_per_s": projected[2],
        "prior_transition3_projected_action470_per_s": projected[3],
        "gauge_fixed_physical_hidden_rate560_per_s": physical_hidden,
        "gauge_fixed_physical_hidden_field_energy_fractions": field_energy,
        "gauge_fixed_physical_hidden_radial_quartile_energy_fractions": radial_energy,
        "anchor_active_coordinates": anchor_active,
    }
    return metrics, arrays


def _checks(metrics: dict) -> dict[str, bool]:
    energy = metrics["hidden_block_energy_fractions"]
    captures = metrics["prior_transition_total_energy_capture"]
    return {
        "macro_restriction_identical": metrics[
            "macro_restriction_bitwise_matches_prior_transition_architecture"
        ],
        "macro_lifting_identical": metrics[
            "macro_lifting_bitwise_matches_prior_transition_architecture"
        ],
        "primary_anchor_identical": metrics[
            "anchor_state_bitwise_matches_primary_candidate"
        ],
        "hidden_fraction_reproduced": abs(
            metrics["hidden_fraction_recomputed"]
            - metrics["hidden_fraction_recorded"]
        )
        <= GEOMETRY_GATE,
        "coordinate_rate_decomposition": metrics[
            "coordinate_rate_decomposition_relative_defect"
        ]
        <= GEOMETRY_GATE,
        "hidden_block_energy_partition": metrics["hidden_block_energy_sum_defect"]
        <= ENERGY_PARTITION_GATE,
        "explicit_amplitudes_have_no_hidden_action": energy[
            "explicit_amplitude2"
        ]
        <= ENERGY_PARTITION_GATE,
        "memory_and_departure_dominate_hidden_action": metrics[
            "memory_plus_departure_hidden_energy_fraction"
        ]
        >= MEMORY_DEPARTURE_ENERGY_MIN,
        "prior_two_mode_transition_interior_rejected": captures["2"]
        <= PRIOR_TWO_MODE_TOTAL_CAPTURE_MAX,
        "prior_three_mode_transition_interior_rejected": captures["3"]
        <= PRIOR_THREE_MODE_TOTAL_CAPTURE_MAX,
        "anchor_outside_prior_active_trust_radius": metrics[
            "anchor_active_radius_over_prior_validated_radius"
        ]
        > 1.0,
        "gauge_fixed_physical_lift": metrics[
            "gauge_fixed_physical_hidden_lift_relative_residual"
        ]
        <= GEOMETRY_GATE,
        "truth_budget": metrics["new_exact_fixed_Q_rate_calls"] == 0,
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_nonlinear_roots"] == 0,
        "propagation_budget": metrics["propagated_states"] == 0,
        "sealed_budget": metrics["sealed_16ms_truth_calls"] == 0,
    }


def _architecture(metrics: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "diagnosis": {
            "fixed_macro_root_at_primary_anchor_rejected": True,
            "physical_fixed_Q_failure_detected": False,
            "primary_anchor_role": "unclassified_transition_sector_candidate",
            "primary_anchor_is_certified_branch_state": False,
            "hidden_rate_fraction": metrics["hidden_fraction_recomputed"],
            "hidden_energy_by_exact_coordinate_block": metrics[
                "hidden_block_energy_fractions"
            ],
            "prior_two_transition_mode_total_capture": metrics[
                "prior_transition_total_energy_capture"
            ]["2"],
            "prior_three_active_mode_total_capture": metrics[
                "prior_transition_total_energy_capture"
            ]["3"],
            "prior_three_coordinate_internal_transition_model_sufficient": False,
        },
        "two_level_hybrid_architecture": {
            "online_branch_state": "s_b=(U80,a2,m_b,branch_label)",
            "online_branch_dynamics": (
                "second_order_conservative_macro_integrator_with_only_"
                "branch_certified_stable_memory"
            ),
            "event_trigger": "branch_fold_or_fast_stability_surface_with_hysteresis",
            "online_transition": (
                "one_prevalidated_conservative_entry_to_exit_impulse_map"
            ),
            "online_transition_ODE": False,
            "online_exact_truth_calls": 0,
            "online_fast_microsteps": 0,
            "offline_transition_reference_state": "full_exact_y470_chart",
            "offline_hidden_state": (
                "eta388=(unresolved_storage80,memory280,departure28)"
            ),
            "offline_reduction": (
                "rank_adaptive_dual_consistent_basis_inside_kernel_of_R82"
            ),
            "offline_full470_fallback_required": True,
            "transition_operator": (
                "T_direction:(s_minus,event_parameters)->"
                "(s_plus,Delta_face_flux,Delta_sources,Delta_MJE,duration,new_branch)"
            ),
            "conservation": (
                "macro_jump_equals_integrated_single_valued_face_flux_plus_"
                "source_and_constraint_work_ledgers"
            ),
        },
        "prospective_hidden_basis_screen": {
            "work_package": AUTHORIZED_NEXT,
            "truth_policy": "saved_arrays_only",
            "new_exact_rate_calls_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_roots_equal": 0,
            "propagated_states_equal": 0,
            "sealed_16ms_truth_calls_equal": 0,
            "coordinate_residual": "H=Qz_F",
            "training_family": (
                "thirteen_revealed_exact_local_rate_snapshots_from_the_prior_"
                "transition_seed"
            ),
            "current_primary_role": "mandatory_heldout_transition_direction",
            "candidate_hidden_ranks": [8, 16, 24, 32, 48, 64, 96, 128],
            "basis_construction": (
                "SVD_of_individually_normalized_hidden_rate_actions_with_"
                "macro_annihilation_preserved_exactly"
            ),
            "physical_structure_audit": (
                "lift_each_candidate_basis_through_the_exact_gauge_fixed_"
                "chart_and_measure_fieldwise_and_radial_capture"
            ),
            "screen_sequence": [
                "fit_only_the_prior_revealed_transition_seed_family",
                "test_capture_of_the_current_exact_primary_hidden_action",
                "if_needed_add_the_primary_direction_as_an_explicit_atlas_center",
                "report_smallest_rank_and_leave_family_out_capture",
                "stop_before_tangent_or_truth_if_rank_exceeds_128",
            ],
            "binding_gates": {
                "macro_annihilation_infinity_max": 5.0e-12,
                "basis_orthonormality_infinity_max": 5.0e-12,
                "training_minimum_hidden_action_energy_capture": 0.99,
                "current_primary_hidden_action_energy_capture": 0.95,
                "current_primary_gauge_fixed_physical_action_energy_capture": 0.95,
                "maximum_selected_hidden_rank": 128,
            },
            "decision": {
                "seed_basis_passes_current_primary": (
                    "common_transition_hidden_basis_candidate_supported_"
                    "definitions_only_tangent_manifest_authorized"
                ),
                "primary_augmentation_passes": (
                    "multi_center_transition_hidden_atlas_required_"
                    "definitions_only_sampling_manifest_authorized"
                ),
                "rank_or_capture_fails": (
                    "transition_internal_reduction_not_supported_"
                    "full470_offline_impulse_map_retained"
                ),
            },
        },
        "authorization_boundaries": {
            "branch_root_authorized": False,
            "complete_tangent_authorized_in_this_package": False,
            "transition_truth_campaign_authorized": False,
            "online_solver_authorized": False,
            "exploratory_cycle_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            "parent_summary": _sha(parent.CANONICAL_DIRECTORY / "summary.json"),
            "parent_rate_arrays": _sha(PARENT_RATE_ARRAYS),
            "parent_rate_metrics": _sha(PARENT_RATE_METRICS),
            "prior_transition_arrays": _sha(TRANSITION_ARRAYS),
            "prior_transition_metrics": _sha(TRANSITION_METRICS),
            "prior_transition_architecture": _sha(TRANSITION_ARCHITECTURE),
            "candidate_arrays": _sha(CANDIDATE_ARRAYS),
            "exact_chart_arrays": _sha(EXACT_CHART_ARRAYS),
            "local_atlas_seed": _sha(LOCAL_ATLAS_SEED),
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
        raise RuntimeError("transition-sector revision manifest already exists")
    metrics, arrays = _reconciliation()
    checks = _checks(metrics)
    if not all(checks.values()):
        raise RuntimeError(f"transition-sector reconciliation failed: {checks}")
    architecture = _architecture(metrics)

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "transition_reconciliation_arrays.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "transition_reconciliation_metrics.json",
        {"metrics": metrics, "checks": checks, "passed": True},
    )
    _write_json(
        CANONICAL_DIRECTORY / "revised_hybrid_architecture.json", architecture
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            **frozen,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "parent_anchor_rejection_preserved": True,
        "physical_fixed_Q_failure_detected": False,
        "primary_anchor_role": "unclassified_transition_sector_candidate",
        "fixed_macro_root_authorized": False,
        "prior_three_coordinate_transition_internal_model_rejected": True,
        "online_hybrid_impulse_map_architecture_preserved": True,
        "offline_full470_reference_preserved": True,
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "online_solver_authorized": False,
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
        transition.THIS_RUNNER,
        transition.THIS_TEST,
        candidate.THIS_RUNNER,
        candidate.THIS_TEST,
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
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
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
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    energy = metrics["hidden_block_energy_fractions"]
    capture = metrics["prior_transition_total_energy_capture"]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Transition-sector macrostate revision WP10c9d6c7c3b5c4f25dh",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The exact 20 ms rate confirms that this anchor is not a nearby frozen-macro critical point. It remains an unclassified transition-sector candidate; this is a closure-coordinate diagnosis, not a physical fixed-Q failure.",
                "",
                f"Hidden-action energy is `{energy['memory280']:.6%}` memory, `{energy['departure28']:.6%}` departure, and `{energy['unresolved_storage80']:.6%}` unresolved storage. The explicit macro amplitudes contain no hidden action.",
                "",
                f"The prior two-coordinate transition sector captures only `{capture['2']:.6%}` of total hidden energy. Even all three old active directions capture only `{capture['3']:.6%}`. The anchor active radius is `{metrics['anchor_active_radius_over_prior_validated_radius']:.3f}` times the old validated radius.",
                "",
                f"In the exact gauge-fixed physical lift, field index 3 carries `{metrics['gauge_fixed_physical_hidden_field_energy_fractions'][3]:.6%}` of the hidden-action energy and the outer radial quartile carries `{metrics['gauge_fixed_physical_hidden_radial_quartile_energy_fractions'][3]:.6%}`. The next basis screen must preserve this physical localization as well as coordinate-space energy.",
                "",
                "The corrected architecture keeps the cheap online hybrid event map but uses the full exact y470 chart as the offline transition reference. Any offline reduction must be rank-adaptive in the dual-consistent hidden kernel and must include memory; the old three-coordinate internal model is rejected.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`, a saved-array-only hidden-basis screen. No new truth, tangent, root, propagation, sealed 16 ms call, online solver, or reduced cycle is authorized.",
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
