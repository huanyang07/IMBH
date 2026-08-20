#!/usr/bin/env python3
"""Freeze the intrinsic hidden-fast branch-root pilot; execute no new truth."""

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

import run_causal_inner_hybrid_candidate_geometry_preflight_wp10c9d6c7c3b5c4f25dc as parent  # noqa: E402
import run_causal_inner_local_slaving_transition_diagnosis_wp10c9d6c7c3b5c4f25da as diagnosis  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dd"
PARENT_COMMIT = "1e00274841d17e382e01c7cec78ffe484572a06a"
PARENT_PARENT = "c94f6cd17fa5273cd0c98ce3d332fcd84480b5b0"
PARENT_TREE = "1160ff33e6ee8ef163f2d508ab2206daf6b994c4"

CLASSIFICATION = (
    "intrinsic_470_hidden_fast_branch_root_pilot_manifest_frozen_"
    "exact_geometric_chart_preflight_authorized"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25de"

FULL_PHYSICAL_DIMENSION = 560
CHART_DIMENSION = 470
MACRO_DIMENSION = 82
HIDDEN_DIMENSION = CHART_DIMENSION - MACRO_DIMENSION
PRIMARY_INDEX = 5
SEALED_INDEX = 4

RESTRICTION_LIFTING_GATE = 5.0e-12
KERNEL_GEOMETRY_GATE = 5.0e-12
DECODER_ERROR_GATE = 5.0e-2
RECONSTRUCTION_GATE = 1.0 - 1.0e-12
HEIGHT_RATIO_GATE = 0.5
OPTICAL_DEPTH_GATE = 1.0

ARTIFACT = (
    "causal_inner_hidden_fast_branch_root_pilot_manifest_"
    "wp10c9d6c7c3b5c4f25dd"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_hidden_fast_branch_root_pilot_manifest_"
    "wp10c9d6c7c3b5c4f25dd.py"
)
THIS_TEST = (
    "tests/test_causal_inner_hidden_fast_branch_root_pilot_manifest_"
    "wp10c9d6c7c3b5c4f25dd.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HIDDEN_FAST_BRANCH_ROOT_PILOT_"
    "MANIFEST_WP10C9D6C7C3B5C4F25DD_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

OLD_HOMOTOPY_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_bordered_branch_homotopy_launch_"
    "wp10c9d6c7c3b5c4f25as"
)
OLD_HESSIAN_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_coordinate_hessian_diagnosis_"
    "wp10c9d6c7c3b5c4f25au"
)


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
        raise RuntimeError("branch-root pilot parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("branch-root pilot parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("branch-root pilot parent tree changed")
    parent_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    parent_summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    parent_contract = _read(
        parent.CANONICAL_DIRECTORY / "branch_pilot_contract.json"
    )
    parent_provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not parent_summary["passed"]
        or parent_summary["classification"] != parent.CLASSIFICATION
        or parent_summary["authorized_next"] != WORK_PACKAGE
        or parent_summary["branch_root_execution_authorized"]
        or parent_contract["next_definitions_only_manifest"]["work_package"]
        != WORK_PACKAGE
    ):
        raise RuntimeError("candidate geometry authorization changed")
    for relative, expected in parent_provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"candidate geometry source changed: {relative}")
    for name, expected in parent_provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")

    diagnosis_hashes = _checksums(diagnosis.CANONICAL_DIRECTORY)
    old_homotopy_hashes = _checksums(OLD_HOMOTOPY_DIRECTORY)
    old_hessian_hashes = _checksums(OLD_HESSIAN_DIRECTORY)
    old_homotopy = _read(OLD_HOMOTOPY_DIRECTORY / "summary.json")
    old_hessian = _read(OLD_HESSIAN_DIRECTORY / "summary.json")
    if (
        old_homotopy["classification"]
        != "bordered_homotopy_launch_failed_conditional_branch_path_requires_diagnosis"
        or old_homotopy["passed"]
        or old_hessian["classification"]
        != "coordinate_hessian_recovery_failed_branch_solver_architecture_requires_revision"
        or old_hessian["passed"]
    ):
        raise RuntimeError("prior branch-root rejection changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("branch-root pilot manifest requires a clean tracked tree")
    return {
        "parent_hashes": parent_hashes,
        "diagnosis_hashes": diagnosis_hashes,
        "old_homotopy_hashes": old_homotopy_hashes,
        "old_hessian_hashes": old_hessian_hashes,
    }


def _selected_decoded_physical_metrics(
    decoded_states: np.ndarray,
    target_coordinates: np.ndarray,
    restriction: np.ndarray,
) -> tuple[dict, np.ndarray]:
    model = parent.field_manifest.ForwardQuadraticAuthenticCenterField(
        _load_npz(parent.FIELD_ARRAYS)
    ).model
    context = model.components["context"]
    audit = (
        parent.field_manifest.vector_field.manifest.parent.geometry.chart_tools._state_audit
    )
    records = []
    reencoded_coordinates = []
    for index in (PRIMARY_INDEX, SEALED_INDEX):
        decoded = np.asarray(decoded_states[index], dtype=float)
        physical = audit(context, decoded)
        reencoded, _ = model.coordinate(decoded)
        target = np.asarray(target_coordinates[index], dtype=float)
        target_macro = restriction @ target
        reencoded_macro = restriction @ reencoded
        record = {
            "candidate_index": index,
            "minimum_reconstruction_factor": float(
                physical["minimum_reconstruction_factor"]
            ),
            "maximum_height_ratio": float(physical["maximum_h_over_r"]),
            "minimum_scattering_optical_depth": float(
                physical["minimum_scattering_optical_depth"]
            ),
            "raw_decoder_coordinate_closure_relative_defect": float(
                np.linalg.norm(reencoded - target)
                / max(
                    np.linalg.norm(reencoded),
                    np.linalg.norm(target),
                    np.finfo(float).tiny,
                )
            ),
            "raw_decoder_macro_closure_infinity": float(
                np.max(np.abs(reencoded_macro - target_macro))
            ),
        }
        record["passed"] = bool(
            record["minimum_reconstruction_factor"] >= RECONSTRUCTION_GATE
            and record["maximum_height_ratio"] <= HEIGHT_RATIO_GATE
            and record["minimum_scattering_optical_depth"] >= OPTICAL_DEPTH_GATE
        )
        records.append(record)
        reencoded_coordinates.append(reencoded)
    metrics = {
        "primary": records[0],
        "sealed": records[1],
        "all_selected_decoded_states_physically_admissible": bool(
            all(record["passed"] for record in records)
        ),
        "raw_decoder_maximum_macro_closure_infinity": float(
            max(record["raw_decoder_macro_closure_infinity"] for record in records)
        ),
        "raw_decoder_is_exact_coordinate_chart": bool(
            max(record["raw_decoder_macro_closure_infinity"] for record in records)
            <= RESTRICTION_LIFTING_GATE
        ),
    }
    return metrics, np.asarray(reencoded_coordinates)


def _fiber_geometry() -> tuple[dict[str, np.ndarray], dict]:
    candidates = _load_npz(
        parent.CANONICAL_DIRECTORY / "candidate_geometry_arrays.npz"
    )
    diagnostic = _load_npz(diagnosis.CANONICAL_DIRECTORY / "diagnostic_arrays.npz")
    restriction = np.asarray(diagnostic["macro_restriction"], dtype=float)
    lifting = np.asarray(
        diagnostic["constraint_compatible_piecewise_constant_lifting"],
        dtype=float,
    )
    if restriction.shape != (MACRO_DIMENSION, CHART_DIMENSION):
        raise RuntimeError("macro restriction dimension changed")
    if lifting.shape != (CHART_DIMENSION, MACRO_DIMENSION):
        raise RuntimeError("macro lifting dimension changed")

    q, _ = np.linalg.qr(restriction.T, mode="complete")
    hidden = np.asarray(q[:, MACRO_DIMENSION:], dtype=float)
    for column in range(hidden.shape[1]):
        pivot = int(np.argmax(np.abs(hidden[:, column])))
        if hidden[pivot, column] < 0.0:
            hidden[:, column] *= -1.0

    coordinates = np.asarray(
        candidates["candidate_absolute_y470_coordinates"], dtype=float
    )
    macro = coordinates @ restriction.T
    residual = coordinates - macro @ lifting.T
    hidden_coordinates = residual @ hidden
    reconstructed = macro @ lifting.T + hidden_coordinates @ hidden.T
    reconstruction_defects = np.linalg.norm(
        reconstructed - coordinates, axis=1
    ) / np.maximum(np.linalg.norm(coordinates, axis=1), np.finfo(float).tiny)

    singular = np.linalg.svd(restriction, compute_uv=False)
    right_inverse_defect = float(
        np.linalg.norm(restriction @ lifting - np.eye(MACRO_DIMENSION), ord=np.inf)
    )
    kernel_annihilation = float(np.linalg.norm(restriction @ hidden, ord=np.inf))
    kernel_orthogonality = float(
        np.linalg.norm(hidden.T @ hidden - np.eye(HIDDEN_DIMENSION), ord=np.inf)
    )
    decoded_physical, selected_reencoded = _selected_decoded_physical_metrics(
        candidates["candidate_decoded_primitive_states"], coordinates, restriction
    )
    metrics = {
        "full_physical_dimension": FULL_PHYSICAL_DIMENSION,
        "chart_dimension": CHART_DIMENSION,
        "macro_dimension": MACRO_DIMENSION,
        "hidden_dimension": HIDDEN_DIMENSION,
        "restriction_rank": int(np.linalg.matrix_rank(restriction)),
        "restriction_condition_number": float(singular[0] / singular[-1]),
        "restriction_lifting_identity_infinity": right_inverse_defect,
        "hidden_basis_annihilation_infinity": kernel_annihilation,
        "hidden_basis_orthogonality_infinity": kernel_orthogonality,
        "maximum_candidate_fiber_reconstruction_relative_defect": float(
            np.max(reconstruction_defects)
        ),
        "primary_decoder_relative_error": float(
            candidates["candidate_decoder_relative_errors"][PRIMARY_INDEX]
        ),
        "sealed_decoder_relative_error": float(
            candidates["candidate_decoder_relative_errors"][SEALED_INDEX]
        ),
        "primary_forward_patch_weight": float(
            candidates["candidate_forward_patch_weights"][PRIMARY_INDEX]
        ),
        "sealed_forward_patch_weight": float(
            candidates["candidate_forward_patch_weights"][SEALED_INDEX]
        ),
        "decoded_physical": decoded_physical,
        "exact_geometric_chart_required": True,
        "primary_and_sealed_labels": "unclassified",
    }
    arrays = {
        "macro_restriction_R82": restriction,
        "macro_lifting_L82": lifting,
        "hidden_orthonormal_basis_Z388": hidden,
        "candidate_macro_coordinates_X82": macro,
        "candidate_hidden_coordinates_z388": hidden_coordinates,
        "candidate_fiber_reconstructed_y470": reconstructed,
        "candidate_fiber_reconstruction_relative_defects": reconstruction_defects,
        "primary_X82": macro[PRIMARY_INDEX],
        "primary_z388": hidden_coordinates[PRIMARY_INDEX],
        "sealed_X82": macro[SEALED_INDEX],
        "sealed_z388": hidden_coordinates[SEALED_INDEX],
        "selected_raw_decoder_reencoded_y470": selected_reencoded,
    }
    return arrays, metrics


def _checks(metrics: dict) -> dict[str, bool]:
    return {
        "restriction_full_rank": metrics["restriction_rank"] == MACRO_DIMENSION,
        "restriction_lifting": metrics["restriction_lifting_identity_infinity"]
        <= RESTRICTION_LIFTING_GATE,
        "hidden_annihilation": metrics["hidden_basis_annihilation_infinity"]
        <= KERNEL_GEOMETRY_GATE,
        "hidden_orthogonality": metrics["hidden_basis_orthogonality_infinity"]
        <= KERNEL_GEOMETRY_GATE,
        "fiber_reconstruction": metrics[
            "maximum_candidate_fiber_reconstruction_relative_defect"
        ]
        <= KERNEL_GEOMETRY_GATE,
        "primary_decoder_geometry": metrics["primary_decoder_relative_error"]
        <= DECODER_ERROR_GATE,
        "sealed_decoder_geometry": metrics["sealed_decoder_relative_error"]
        <= DECODER_ERROR_GATE,
        "decoded_physical": metrics["decoded_physical"][
            "all_selected_decoded_states_physically_admissible"
        ],
        "raw_decoder_mismatch_detected": not metrics["decoded_physical"][
            "raw_decoder_is_exact_coordinate_chart"
        ],
        "exact_geometric_chart_required": metrics[
            "exact_geometric_chart_required"
        ],
        "old_invalid_field_not_reused": metrics["primary_forward_patch_weight"] == 0.0
        and metrics["sealed_forward_patch_weight"] == 0.0,
        "candidates_unclassified": metrics["primary_and_sealed_labels"]
        == "unclassified",
        "no_truth": True,
        "no_root": True,
        "no_propagation": True,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "mathematical_objects": {
            "physical_truth_state": "x_in_R560",
            "coordinate_map": "y=C470(x)_in_R470",
            "online_macro_state": "X=R82_y_in_R82",
            "hidden_candidate_coordinate": "z=Z388_transpose_(y-L82_X)",
            "candidate_fiber": "y(X,z)=L82_X+Z388_z",
            "raw_approximate_decoder": "D0_470(y)_used_only_as_initial_guess",
            "exact_geometric_chart": (
                "chi_xstar(y)_solves_C470(x)=y_and_N90_transpose_W_"
                "(x-xstar)=0"
            ),
            "exact_chart_dimension": "470_coordinate_equations_plus_90_gauge_equations",
            "candidate_hidden_residual": (
                "H_X(z)=Z388_transpose_DC470_at_chi(y)_times_fQ(chi(y))"
            ),
            "physical_vector_field": "certified_exact_fixed_Q_field_fQ(x)",
            "coordinate_field": "F470(y)=DC470_at_chi(y)_times_fQ(chi(y))",
            "no_artificial_82_channel_physical_reaction": True,
        },
        "truth_hierarchy": {
            "candidate_only": (
                "raw_decoder_initial_guess_local_surrogate_and_Broyden_iterate"
            ),
            "binding": (
                "exact_geometric_chart_state_exact_fixed_Q_rate_complete_"
                "coordinate_tangent_and_post_root_physical_audit"
            ),
            "old_forward_patch_may_seed_this_root": False,
            "reason": "both_selected_candidates_have_zero_validated_patch_weight",
            "decoder_normal_component_is_binding": True,
            "raw_decoder_macro_mismatch_may_be_ignored": False,
        },
        "preserved_negative_results": {
            "old_560_state_bordered_KKT_homotopy": (
                "failed_and_may_not_be_reinterpreted_or_reused_as_a_pass"
            ),
            "old_coordinate_Hessian_recovery": (
                "failed_due_to_complete_conditioning_and_may_not_be_omitted"
            ),
            "new_distinction": (
                "exact_implicit_470_chart_intrinsic_Z388_hidden_residual_"
                "separate_physical_validation_and_no_multiplier_KKT"
            ),
        },
        "prospective_execution": {
            "work_package": AUTHORIZED_NEXT,
            "purpose": "exact_geometric_470_chart_preflight_only",
            "candidate": "U20_unclassified_primary",
            "sealed_candidate": "U16_unclassified_sealed",
            "sealed_candidate_truth_calls_equal": 0,
            "stage_order": [
                "construct_state_local_C470_Jacobian_and_N90_gauge_at_accepted_20ms_state",
                "prove_anchor_roundtrip_and_coordinate_rank",
                "retract_prospectively_frozen_small_hidden_and_macro_directions",
                "require_exact_coordinate_closure_and_all_physical_guards",
                "audit_exact_chart_directional_derivative_and_conditioning",
            ],
            "budgets": {
                "new_exact_fixed_Q_rate_evaluations_equal": 0,
                "new_complete_physical_generator_assemblies_equal": 0,
                "new_intrinsic_hidden_roots_equal": 0,
                "coordinate_retractions_max": 18,
                "propagated_states_equal": 0,
                "transition_executions_equal": 0,
            },
            "exact_chart": {
                "anchor": "accepted_20ms_physical_state_not_raw_decoded_state",
                "equations": "C470(x)-y_equal_zero_and_N90_transpose_W_(x-xstar)_equal_zero",
                "initial_guess": "raw_decoder_then_previous_retracted_neighbor",
                "nonlinear_tolerance_infinity": 1.0e-10,
                "maximum_scaled_physical_departure": 0.015,
                "maximum_condition_number": 1.0e7,
                "directional_derivative_relative_defect_max": 1.0e-6,
                "no_post_retraction_projection": True,
            },
            "direction_design": {
                "hidden_directions": 4,
                "macro_directions": 4,
                "signed_radius": 0.0025,
                "directions_frozen_before_retraction": True,
                "sealed_16ms_not_used_to_select_directions": True,
            },
        },
        "exact_chart_preflight_gates": {
            "anchor_roundtrip_bitwise_or_zero_correction": True,
            "coordinate_rank_equal": 470,
            "coordinate_closure_infinity_max": 1.0e-10,
            "gauge_closure_infinity_max": 1.0e-10,
            "maximum_condition_number": 1.0e7,
            "directional_derivative_relative_defect_max": 1.0e-6,
            "minimum_reconstruction_factor": RECONSTRUCTION_GATE,
            "maximum_height_ratio": HEIGHT_RATIO_GATE,
            "minimum_scattering_optical_depth": OPTICAL_DEPTH_GATE,
            "all_18_retractions_must_pass": True,
            "failure_action": "stop_without_any_rate_call_or_branch_root",
        },
        "future_root_contract_frozen_but_not_authorized": {
            "work_package_after_chart_pass": "WP10c9d6c7c3b5c4f25df",
            "candidate": "U20_unclassified_primary",
            "root": "H_Xstar(z)=0_through_exact_chart_chi_xstar",
            "new_exact_fixed_Q_rate_evaluations_max": 12,
            "new_complete_physical_generator_assemblies_max": 2,
            "new_intrinsic_hidden_roots_max": 1,
            "initial_hidden_rate_relative_fraction_max": 0.25,
            "complete_coordinate_tangent_JVP_relative_defect_max": 1.0e-6,
            "equilibrated_hidden_condition_number_max": 1.0e8,
            "linear_predictor_maximum_scaled_physical_component_max": 0.015,
            "normalized_hidden_residual_infinity_max": 1.0e-10,
            "exact_coordinate_closure_infinity_max": 1.0e-10,
            "decoder_normal_rate_relative_defect_max": 0.10,
            "slow_graph_invariance_relative_defect_max": 0.10,
            "fast_spectral_abscissa_max_per_second": 0.0,
            "spectral_gap_ratio_min": 10.0,
            "coordinate_Hessian_term_required": True,
            "authorization_requires_chart_preflight_pass": True,
        },
        "decision": {
            "stable_root_pass": {
                "classification": "primary_local_stable_branch_seed_supported_unclassified",
                "authorizes_only": (
                    "definitions_only_one_direction_pseudo_arclength_"
                    "continuation_manifest"
                ),
                "cold_or_hot_label_assigned": False,
            },
            "converged_but_unstable": {
                "classification": "primary_stationary_transition_or_fold_marker_not_slow_branch",
                "authorizes_only": "definitions_only_transition_capture_manifest",
            },
            "preflight_or_root_failure": {
                "classification": "primary_local_branch_seed_not_established",
                "authorizes_only": "definitions_only_candidate_reselection_or_transition_manifest",
            },
            "exact_validation_failure": {
                "classification": "470_chart_branch_candidate_rejected_by_physical_truth",
                "authorizes_only": "definitions_only_decoder_or_explicit_state_expansion_manifest",
            },
            "sealed_16ms_opened_by_any_outcome": False,
        },
        "authorization_boundaries": {
            "this_package_definitions_only": True,
            "new_truth_in_this_package": False,
            "branch_root_in_this_package": False,
            "branch_root_in_next_package": False,
            "exact_chart_preflight_in_next_package": True,
            "online_solver_authorized": False,
            "physical_microburst_authorized": False,
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
        raise RuntimeError("hidden-fast branch-root pilot manifest already exists")
    arrays, metrics = _fiber_geometry()
    checks = _checks(metrics)
    if not all(checks.values()):
        raise RuntimeError(f"hidden-fast branch-root geometry failed: {checks}")
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "fiber_geometry.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "fiber_geometry_metrics.json",
        {
            "metrics": metrics,
            "checks": checks,
            "passed": True,
            "new_exact_rate_calls": 0,
            "new_complete_generator_assemblies": 0,
            "new_nonlinear_roots": 0,
            "propagated_states": 0,
        },
    )
    _write_json(CANONICAL_DIRECTORY / "branch_root_pilot_contract.json", contract)
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
        "full_physical_dimension": FULL_PHYSICAL_DIMENSION,
        "chart_dimension": CHART_DIMENSION,
        "macro_dimension": MACRO_DIMENSION,
        "hidden_dimension": HIDDEN_DIMENSION,
        "selected_decoded_states_physically_admissible": True,
        "raw_decoder_is_exact_coordinate_chart": False,
        "exact_geometric_chart_preflight_authorized": True,
        "old_failed_560_KKT_not_reused": True,
        "old_zero_weight_forward_patch_not_reused": True,
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "branch_root_execution_authorized": False,
        "sealed_16ms_execution_authorized": False,
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
        diagnosis.THIS_RUNNER,
        diagnosis.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
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
                "# Hidden-fast branch-root pilot manifest WP10c9d6c7c3b5c4f25dd",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The conservative U80+a2 restriction has rank 82, its frozen lifting closes to machine precision, and the orthonormal hidden fiber has dimension 388. The selected decoded 20 ms primary and sealed 16 ms states both pass reconstruction, height, and optical-depth guards.",
                "",
                f"The raw approximate decoder is physically admissible but is not an exact chart here: its selected-state macro closure mismatch reaches `{metrics['decoded_physical']['raw_decoder_maximum_macro_closure_infinity']:.6e}` in normalized coordinates. It is therefore restricted to initial-guess duty.",
                "",
                "The replacement is an implicit exact geometric chart: 470 coordinate equations plus a 90-dimensional gauge define a unique physical state. The old failed 560-state multiplier KKT/homotopy is not retried, and no artificial 82-channel physical reaction is introduced.",
                "",
                "The next work package may only construct and validate this exact chart around the accepted 20 ms state using zero fixed-Q rate calls, zero generator assemblies, and zero branch roots. The 16 ms candidate remains sealed. A root may be authorized only after all 18 prospective retractions close and pass physical guards.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`.",
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
