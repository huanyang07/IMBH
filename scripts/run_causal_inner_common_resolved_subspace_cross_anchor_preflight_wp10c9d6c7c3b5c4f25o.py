#!/usr/bin/env python3
"""Execute the two-anchor common-resolved-subspace preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import null_space


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_common_resolved_subspace_cross_anchor_manifest_wp10c9d6c7c3b5c4f25n as manifest  # noqa: E402
import run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1 as history  # noqa: E402
import run_causal_inner_face36_leading_two_plus_hmm_fixed_q_preflight_wp10c9d6c7c3b5c4f22 as mode_tools  # noqa: E402
import run_causal_inner_invariant_projection_spectrum_audit_wp10c9d6c7c3b5c4f25e as projection_tools  # noqa: E402
import run_causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k as r32_tools  # noqa: E402
import run_causal_inner_pathwise_closure_descriptor_pilot_wp10c9d6c7c3b5c4f25c as generator_tools  # noqa: E402
import run_causal_inner_resolved_mode_promotion_audit_wp10c9d6c7c3b5c4f25g as promotion_tools  # noqa: E402
import run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m as memory_tools  # noqa: E402

from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    _q3_physical_selectors,
    causal_five_field_fixed_q_reaction_jvp,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    _descriptor_matrices,
    _node_reconstruction_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25o"
MANIFEST_COMMIT = "85b48069360e5419dd91233bfef70034ba73308e"
MANIFEST_PARENT = "a23a39c52756b93dd30b2c988965347b1417c746"
MANIFEST_TREE = "ebae2cfbf85e8a0d535709260861b69527b9cfc2"

ARTIFACT = "causal_inner_common_resolved_subspace_cross_anchor_preflight_wp10c9d6c7c3b5c4f25o"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_common_resolved_subspace_cross_anchor_preflight_wp10c9d6c7c3b5c4f25o.py"
THIS_TEST = "tests/test_causal_inner_common_resolved_subspace_cross_anchor_preflight_wp10c9d6c7c3b5c4f25o.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COMMON_RESOLVED_SUBSPACE_"
    "CROSS_ANCHOR_PREFLIGHT_WP10C9D6C7C3B5C4F25O_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
CONSERVATIVE_FIELDS = (0, 2, 3)

GLOBAL_PASS_CLASSIFICATION = (
    "two_anchor_common_subspace_R96_memory_passed_"
    "online_prototype_manifest_authorized"
)
ATLAS_PASS_CLASSIFICATION = (
    "two_anchor_local_models_passed_common_chart_failed_"
    "conservative_atlas_manifest_authorized"
)
MEMORY_FAIL_CLASSIFICATION = (
    "heldout_R32_R96_memory_failed_architecture_reassessment_required"
)
NUMERICAL_FAIL_CLASSIFICATION = (
    "heldout_generator_or_common_projection_numerical_failure_stop"
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
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
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = _sha(directory / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = actual
    return recorded


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(float).tiny)
    )


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("cross-anchor manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("cross-anchor manifest parent changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("cross-anchor manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["heldout_16ms_generator_preflight_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["execution_budget"]["allowed_new_full_560_direction_generator_assemblies"] != 1
        or contract["common_basis_memory"]["order"] != manifest.MEMORY_ORDER
    ):
        raise RuntimeError("cross-anchor execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    decisive = {
        "selected_R32_memory_models": manifest.PARENT_DIRECTORY / "candidate_models.npz",
        "selected_R32_memory_errors": manifest.PARENT_DIRECTORY / "candidate_errors.npz",
        "selected_R32_memory_metrics": manifest.PARENT_DIRECTORY / "metrics.json",
        "primary_R32_projection_promotion": manifest.R32_DIRECTORY / "R32_projection_promotion.npz",
        "primary_R32_transfer": manifest.R32_DIRECTORY / "R32_transfer.npz",
        "primary_complete_fixed_Q_generator": manifest.GENERATOR_DIRECTORY / "descriptor_A.npz",
        "middle_6ms_arrays": manifest.MIDDLE_PILOT_ARRAYS,
        "middle_20ms_arrays": manifest.MIDDLE_ARRAYS,
    }
    for name, path in decisive.items():
        if _sha(path) != contract["parent_decisive_hashes"][name]:
            raise RuntimeError(f"cross-anchor decisive input changed: {path}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("cross-anchor execution requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _heldout_output_map(data: dict) -> tuple[np.ndarray, np.ndarray]:
    boundaries = generator_tools._coarse_boundaries(data["state"].shape[0])
    shared = np.asarray(
        data["tangent"].spatial_tangent.shared_face_flux_scaled_jacobians,
        dtype=float,
    )
    output = shared[boundaries][:, np.asarray(CONSERVATIVE_FIELDS)].reshape(
        -1, data["state"].size
    )
    return output, boundaries


def _assembly_stage() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if SCRATCH_DIRECTORY.exists():
        raise RuntimeError("cross-anchor scratch directory already exists")
    SCRATCH_DIRECTORY.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    data = history._state_data("heldout_16ms")
    state = np.asarray(data["state"], dtype=float)
    columns = np.asarray(data["columns"], dtype=float)
    rows = np.asarray(data["rows"], dtype=float)
    reaction = data["reaction"]
    tangent = data["tangent"]
    dimensions = int(state.size)
    full_jvp = causal_five_field_fixed_q_reaction_jvp(
        data["context"],
        state,
        np.eye(dimensions),
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        reaction=reaction,
    )
    complete, fixed_rate, multiplier_jacobian = generator_tools._continuous_generator(
        tangent.scaled_generator_per_s,
        tangent.scaled_base_rate_per_s,
        reaction.q3_scaled_derivative,
        reaction.reaction_lift,
        full_jvp.q3_scaled_row_derivatives,
        full_jvp.reaction_lift_derivatives,
    )
    saved = np.asarray(mode_tools._saved_directions("middle")[:2], dtype=float)
    leading_scaled = saved.reshape(2, -1).T / columns.ravel()[:, None]
    leading_fixed, _ = mode_tools._physical_reaction_projection(
        leading_scaled,
        reaction.q3_scaled_derivative,
        reaction.reaction_lift,
    )
    duals = mode_tools._stable_a2_duals(
        reaction.descriptor_scaled_matrix,
        reaction.q3_scaled_derivative,
        reaction.reaction_lift,
        leading_fixed,
    )
    audit_direction = leading_fixed[:, 0] / np.linalg.norm(leading_fixed[:, 0])
    directional = causal_five_field_fixed_q_reaction_jvp(
        data["context"],
        state,
        audit_direction,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        reaction=reaction,
    )
    direct_action = generator_tools._continuous_generator_action(
        tangent.scaled_generator_per_s @ audit_direction,
        tangent.scaled_base_rate_per_s,
        reaction.q3_scaled_derivative,
        reaction.reaction_lift,
        directional.q3_scaled_row_derivatives[0],
        directional.reaction_lift_derivatives[0],
    )
    jvp_defect = _relative(direct_action, complete @ audit_direction)
    constraint_differential = np.empty((3, dimensions))
    for column in range(dimensions):
        constraint_differential[:, column] = (
            reaction.q3_scaled_derivative @ complete[:, column]
            + full_jvp.q3_scaled_row_derivatives[column] @ fixed_rate
        )
    differential_defect = float(
        np.linalg.norm(constraint_differential)
        / max(float(np.linalg.norm(complete)), 1.0)
    )
    output_map, boundaries = _heldout_output_map(data)
    elapsed = float(time.perf_counter() - began)
    gates = frozen["contract"]["heldout_generator"]["pass_requires"]
    metrics = {
        "stage": "heldout_16ms_complete_fixed_Q_generator",
        "state_label": "heldout_16ms",
        "state_time_seconds": data["time_seconds"],
        "truth_dimension": dimensions,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 1,
        "new_truth_anchors": 1,
        "complete_JVP_relative_defect": jvp_defect,
        "constraint_differential_identity_relative_defect": differential_defect,
        "reaction_identity_directional_defect": full_jvp.maximum_identity_directional_defect,
        "reaction_ledger_directional_relative_defect": full_jvp.maximum_reaction_ledger_directional_relative_defect,
        "raw_Schur_condition_number": reaction.raw_schur_condition_number,
        "a2_dual_biorthogonality_defect": duals["qr_metrics"]["biorthogonality_defect"],
        "a2_dual_reaction_annihilation_defect": duals["qr_metrics"][
            "normalized_reaction_annihilation_defect"
        ],
        "wall_seconds": elapsed,
    }
    _write_npz(
        SCRATCH_DIRECTORY / "heldout_generator.npz",
        primitive_state=state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        complete_fixed_Q_generator=complete,
        fixed_Q_rate=fixed_rate,
        multiplier_jacobian=multiplier_jacobian,
        descriptor=np.asarray(reaction.descriptor_scaled_matrix, dtype=float),
        q3_scaled_derivative=np.asarray(reaction.q3_scaled_derivative, dtype=float),
        reaction_lift=np.asarray(reaction.reaction_lift, dtype=float),
        a2_dual=np.asarray(duals["dual_qr"], dtype=float),
        leading_fixed_Q_lifts=leading_fixed,
        output_map=output_map,
        output_face_indices=boundaries,
    )
    roundtrip = True
    with np.load(SCRATCH_DIRECTORY / "heldout_generator.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["primitive_state"], state)
        roundtrip &= np.array_equal(source["complete_fixed_Q_generator"], complete)
        roundtrip &= np.array_equal(source["a2_dual"], duals["dual_qr"])
        roundtrip &= np.array_equal(source["output_map"], output_map)
    metrics["generator_and_state_database_roundtrip_bitwise"] = bool(roundtrip)
    metrics["passed"] = bool(
        metrics["truth_dimension"] == manifest.TRUTH_DIMENSION
        and metrics["new_nonlinear_roots"] == 0
        and metrics["propagated_states"] == 0
        and metrics["new_full_560_direction_generator_assemblies"] == 1
        and metrics["complete_JVP_relative_defect"] <= gates["complete_JVP_relative_defect_max"]
        and metrics["constraint_differential_identity_relative_defect"]
        <= gates["constraint_differential_identity_relative_defect_max"]
        and metrics["reaction_ledger_directional_relative_defect"]
        <= gates["reaction_ledger_directional_relative_defect_max"]
        and metrics["reaction_identity_directional_defect"]
        <= gates["reaction_identity_directional_defect_max"]
        and metrics["raw_Schur_condition_number"] <= gates["maximum_raw_Schur_condition_number"]
        and metrics["generator_and_state_database_roundtrip_bitwise"]
        and metrics["wall_seconds"]
        <= 3600.0 * frozen["contract"]["execution_budget"]["maximum_wall_hours"]
    )
    _write_json(SCRATCH_DIRECTORY / "assembly_metrics.json", metrics)
    return metrics


def _local_projection(
    generator: np.ndarray,
    state: np.ndarray,
    columns: np.ndarray,
    rows: np.ndarray,
    context,
    a2_dual: np.ndarray,
    output_map: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    node_weights, node_cells, node_radii, node_measures, reconstruction_defect, partition_defect = _node_reconstruction_weights(
        context, state
    )
    mapped, height = _descriptor_matrices(
        context,
        state,
        columns,
        rows,
        node_weights,
        node_cells,
        node_radii,
        node_measures,
    )
    storage, physical_storage, storage_scales = r32_tools._R32_storage_restriction(
        mapped, height, rows
    )
    restriction, lifting, complement, projection_metrics = projection_tools._complete_qr_projection(
        storage, a2_dual
    )
    q_selectors = _q3_physical_selectors(state.shape[0], 72, rows)
    q_physical = np.asarray(q_selectors @ mapped, dtype=float)
    q_scaled = q_physical / np.linalg.norm(q_physical, axis=1)[:, None]
    constraint_defect = _relative(q_scaled @ lifting @ restriction, q_scaled)
    physical_mapped = C * rows.ravel()[:, None] * mapped
    truth_totals = []
    coarse_totals = []
    for field in CONSERVATIVE_FIELDS:
        truth_totals.append(np.sum(physical_mapped[field::5], axis=0))
        coarse_totals.append(np.sum(physical_storage[field::5], axis=0))
    telescope_defect = _relative(np.asarray(coarse_totals), np.asarray(truth_totals))
    promotion, promotion_metrics = promotion_tools._ordered_real_schur_promotion(
        generator,
        restriction,
        lifting,
        complement,
        stability_margin=manifest.STABILITY_MARGIN_PER_SECOND,
    )
    arrays = {
        "storage": storage,
        "physical_storage": physical_storage,
        "storage_scales": storage_scales,
        "restriction": restriction,
        "lifting": lifting,
        "complement": complement,
        "promoted_basis": promotion["promoted_truth_basis"],
        "output_map": output_map,
        "mapped": mapped,
        "height": height,
    }
    metrics = {
        **projection_metrics,
        **promotion_metrics,
        "constraint_rowspace_relative_defect": constraint_defect,
        "M_J_E_telescope_relative_defect": telescope_defect,
        "node_reconstruction_relative_defect": reconstruction_defect,
        "node_partition_of_unity_defect": partition_defect,
    }
    return arrays, metrics


def _common_reference_basis(
    primary_promoted: np.ndarray,
    heldout_promoted: np.ndarray,
    relative_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    combined = np.column_stack((primary_promoted, heldout_promoted))
    left, singular, _ = np.linalg.svd(combined, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * relative_cutoff))
    return left[:, :rank], singular, rank


def _anchor_common_basis(
    reference_basis: np.ndarray,
    complement: np.ndarray,
    lifting: np.ndarray,
    restriction: np.ndarray,
    promoted_basis: np.ndarray,
    generator: np.ndarray,
    output_map: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    projected = complement @ (complement.T @ reference_basis)
    local_basis, triangular = np.linalg.qr(projected, mode="reduced")
    if np.linalg.matrix_rank(triangular) != reference_basis.shape[1]:
        raise RuntimeError("common reference loses rank in an anchor complement")
    left, cosines, right_h = np.linalg.svd(local_basis.T @ reference_basis)
    aligned = local_basis @ left @ right_h
    augmented_restriction = np.vstack((restriction, aligned.T))
    augmented_lifting = np.column_stack((lifting, aligned))
    local_coordinates = aligned.T @ complement
    stable_coordinates = null_space(local_coordinates)
    stable_basis = complement @ stable_coordinates
    stable_operator = stable_basis.T @ generator @ stable_basis
    stable_forcing = stable_basis.T @ generator @ augmented_lifting
    stable_observation = output_map @ stable_basis
    direct = output_map @ augmented_lifting
    capture_defect = float(
        np.linalg.norm(promoted_basis - aligned @ (aligned.T @ promoted_basis))
        / max(float(np.linalg.norm(promoted_basis)), np.finfo(float).tiny)
    )
    stable_poles = np.linalg.eigvals(stable_operator)
    metrics = {
        "common_promoted_dimension": int(aligned.shape[1]),
        "common_online_continuous_dimension": int(
            restriction.shape[0] + aligned.shape[1] + manifest.MEMORY_ORDER
        ),
        "reference_projection_minimum_principal_cosine": float(np.min(cosines)),
        "local_promoted_subspace_projection_relative_defect": capture_defect,
        "common_modal_basis_orthogonality_defect": float(
            np.max(np.abs(aligned.T @ aligned - np.eye(aligned.shape[1])))
        ),
        "common_modal_basis_physical_restriction_defect": float(
            np.max(np.abs(restriction @ aligned))
        ),
        "common_modal_basis_physical_lifting_defect": float(
            np.max(np.abs(aligned.T @ lifting))
        ),
        "common_augmented_restriction_lifting_identity_defect": float(
            np.max(
                np.abs(
                    augmented_restriction @ augmented_lifting
                    - np.eye(augmented_restriction.shape[0])
                )
            )
        ),
        "common_augmented_restriction_stable_annihilation_defect": float(
            np.max(np.abs(augmented_restriction @ stable_basis))
        ),
        "common_stable_basis_orthogonality_defect": float(
            np.max(np.abs(stable_basis.T @ stable_basis - np.eye(stable_basis.shape[1])))
        ),
        "remaining_common_unresolved_dimension": int(stable_basis.shape[1]),
        "remaining_common_unresolved_spectral_abscissa_per_second": float(
            np.max(np.real(stable_poles))
        ),
    }
    return {
        "aligned_common_basis": aligned,
        "augmented_restriction": augmented_restriction,
        "augmented_lifting": augmented_lifting,
        "stable_basis": stable_basis,
        "stable_operator": stable_operator,
        "stable_forcing": stable_forcing,
        "stable_observation": stable_observation,
        "direct": direct,
    }, metrics


def _transfer_and_memory(
    arrays: dict[str, np.ndarray],
    frequencies: np.ndarray,
    heldout_frequencies: np.ndarray,
    gates: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    operator = arrays["stable_operator"]
    forcing = arrays["stable_forcing"]
    observation = arrays["stable_observation"]
    direct = arrays["direct"]
    normalized_forcing, normalized_observation, normalized_direct, input_scales, output_scales = memory_tools._normalize_system(
        forcing, observation, direct
    )
    training_reference, training_reference_residual = memory_tools._frequency_response(
        operator,
        normalized_forcing,
        normalized_observation,
        normalized_direct,
        frequencies,
    )
    heldout_reference, heldout_reference_residual = memory_tools._frequency_response(
        operator,
        normalized_forcing,
        normalized_observation,
        normalized_direct,
        heldout_frequencies,
    )
    balanced, full_metrics = memory_tools._balanced_realization(
        operator, normalized_forcing, normalized_observation
    )
    reduced_operator, reduced_forcing, reduced_observation, truncation = memory_tools._truncate_balanced(
        operator,
        normalized_forcing,
        normalized_observation,
        balanced,
        manifest.MEMORY_ORDER,
    )
    training_approximation, training_reduced_residual = memory_tools._frequency_response(
        reduced_operator,
        reduced_forcing,
        reduced_observation,
        normalized_direct,
        frequencies,
    )
    heldout_approximation, heldout_reduced_residual = memory_tools._frequency_response(
        reduced_operator,
        reduced_forcing,
        reduced_observation,
        normalized_direct,
        heldout_frequencies,
    )
    training_dynamic, training_total, training_errors = memory_tools._error_metrics(
        training_approximation, training_reference, normalized_direct
    )
    heldout_dynamic, heldout_total, heldout_errors = memory_tools._error_metrics(
        heldout_approximation, heldout_reference, normalized_direct
    )
    stability = memory_tools._stability_metrics(reduced_operator)
    metrics = {
        **full_metrics,
        **truncation,
        **stability,
        **{f"training_{key}": value for key, value in training_errors.items()},
        **{f"heldout_{key}": value for key, value in heldout_errors.items()},
        "maximum_frequency_solve_relative_residual": max(
            training_reference_residual,
            heldout_reference_residual,
            training_reduced_residual,
            heldout_reduced_residual,
        ),
    }
    metrics["training_passed"] = memory_tools._gates_pass(metrics, gates, "training")
    metrics["heldout_passed"] = memory_tools._gates_pass(metrics, gates, "heldout")
    metrics["passed"] = bool(
        metrics["training_passed"]
        and metrics["heldout_passed"]
        and metrics["controllability_gramian_relative_residual"] <= 1.0e-8
        and metrics["observability_gramian_relative_residual"] <= 1.0e-8
        and metrics["maximum_frequency_solve_relative_residual"] <= 1.0e-10
    )
    return {
        "input_scales": input_scales,
        "output_scales": output_scales,
        "normalized_direct": normalized_direct,
        "reduced_operator": reduced_operator,
        "reduced_forcing": reduced_forcing,
        "reduced_observation": reduced_observation,
        "training_reference": training_reference,
        "heldout_reference": heldout_reference,
        "training_approximation": training_approximation,
        "heldout_approximation": heldout_approximation,
        "training_dynamic_errors": training_dynamic,
        "training_total_errors": training_total,
        "heldout_dynamic_errors": heldout_dynamic,
        "heldout_total_errors": heldout_total,
        "hankel_singular_values": balanced["hankel_singular_values"],
    }, metrics


def _common_stage() -> dict:
    frozen = _validate_manifest(require_clean=True)
    assembly = _read(SCRATCH_DIRECTORY / "assembly_metrics.json")
    if not assembly["passed"]:
        raise RuntimeError("heldout generator failed; common stage is blocked")
    began = time.perf_counter()
    primary_data = history._state_data("primary_20ms")
    heldout_data = history._state_data("heldout_16ms")
    with np.load(manifest.GENERATOR_DIRECTORY / "descriptor_A.npz", allow_pickle=False) as source:
        primary_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    with np.load(manifest.GENERATOR_DIRECTORY / "projection.npz", allow_pickle=False) as source:
        primary_a2 = np.asarray(source["a2_dual"], dtype=float)
        primary_output = np.asarray(source["output_map"], dtype=float)
    with np.load(SCRATCH_DIRECTORY / "heldout_generator.npz", allow_pickle=False) as source:
        heldout_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
        heldout_a2 = np.asarray(source["a2_dual"], dtype=float)
        heldout_output = np.asarray(source["output_map"], dtype=float)
    primary_local, primary_projection_metrics = _local_projection(
        primary_generator,
        np.asarray(primary_data["state"], dtype=float),
        np.asarray(primary_data["columns"], dtype=float),
        np.asarray(primary_data["rows"], dtype=float),
        primary_data["context"],
        primary_a2,
        primary_output,
    )
    heldout_local, heldout_projection_metrics = _local_projection(
        heldout_generator,
        np.asarray(heldout_data["state"], dtype=float),
        np.asarray(heldout_data["columns"], dtype=float),
        np.asarray(heldout_data["rows"], dtype=float),
        heldout_data["context"],
        heldout_a2,
        heldout_output,
    )
    cutoff = frozen["contract"]["common_resolved_subspace"][
        "union_numerical_rank_relative_cutoff"
    ]
    reference_basis, union_singular, union_dimension = _common_reference_basis(
        primary_local["promoted_basis"],
        heldout_local["promoted_basis"],
        cutoff,
    )
    primary_common, primary_common_metrics = _anchor_common_basis(
        reference_basis,
        primary_local["complement"],
        primary_local["lifting"],
        primary_local["restriction"],
        primary_local["promoted_basis"],
        primary_generator,
        primary_output,
    )
    heldout_common, heldout_common_metrics = _anchor_common_basis(
        reference_basis,
        heldout_local["complement"],
        heldout_local["lifting"],
        heldout_local["restriction"],
        heldout_local["promoted_basis"],
        heldout_generator,
        heldout_output,
    )
    anchor_cosines = np.linalg.svd(
        primary_common["aligned_common_basis"].T
        @ heldout_common["aligned_common_basis"],
        compute_uv=False,
    )
    with np.load(manifest.R32_DIRECTORY / "R32_transfer.npz", allow_pickle=False) as source:
        frequencies = np.asarray(source["angular_frequencies_per_second"], dtype=float)
    heldout_frequencies = memory_tools._heldout_frequencies(frequencies)
    memory_gates = frozen["contract"]["common_basis_memory"][
        "pass_requires_at_each_anchor_on_training_and_heldout"
    ]
    primary_memory, primary_memory_metrics = _transfer_and_memory(
        primary_common, frequencies, heldout_frequencies, memory_gates
    )
    heldout_memory, heldout_memory_metrics = _transfer_and_memory(
        heldout_common, frequencies, heldout_frequencies, memory_gates
    )
    projection_gates = frozen["contract"]["anchor_local_R32_projection"]["pass_requires"]
    common_gates = frozen["contract"]["common_resolved_subspace"]["pass_requires"]

    def projection_pass(metrics: dict) -> bool:
        return bool(
            metrics["resolved_rank"] == projection_gates["resolved_rank"]
            and metrics["resolved_condition_number"] <= projection_gates["resolved_condition_number_max"]
            and metrics["restriction_lifting_identity_defect"] <= projection_gates["restriction_lifting_identity_max"]
            and metrics["restriction_complement_annihilation_defect"] <= projection_gates["restriction_complement_annihilation_max"]
            and metrics["complement_orthogonality_defect"] <= projection_gates["complement_orthogonality_max"]
            and metrics["constraint_rowspace_relative_defect"] <= projection_gates["constraint_rowspace_relative_defect_max"]
            and metrics["M_J_E_telescope_relative_defect"] <= projection_gates["M_J_E_telescope_relative_defect_max"]
            and metrics["remaining_unresolved_spectral_abscissa_per_second"]
            <= projection_gates["remaining_unresolved_spectral_abscissa_per_second_max"]
        )

    def common_pass(metrics: dict) -> bool:
        return bool(
            metrics["local_promoted_subspace_projection_relative_defect"]
            <= common_gates["local_promoted_subspace_projection_relative_defect_max"]
            and metrics["common_modal_basis_orthogonality_defect"]
            <= common_gates["common_modal_basis_orthogonality_defect_max"]
            and metrics["common_augmented_restriction_lifting_identity_defect"]
            <= common_gates["common_augmented_restriction_lifting_identity_defect_max"]
            and metrics["common_augmented_restriction_stable_annihilation_defect"]
            <= common_gates["common_augmented_restriction_stable_annihilation_defect_max"]
            and metrics["remaining_common_unresolved_spectral_abscissa_per_second"]
            <= common_gates["remaining_common_unresolved_spectral_abscissa_per_second_max"]
            and metrics["common_online_continuous_dimension"]
            <= common_gates["maximum_online_continuous_dimension"]
        )

    primary_projection_passed = projection_pass(primary_projection_metrics)
    heldout_projection_passed = projection_pass(heldout_projection_metrics)
    primary_common_passed = common_pass(primary_common_metrics)
    heldout_common_passed = common_pass(heldout_common_metrics)
    dimension_passed = bool(
        union_dimension <= manifest.MAXIMUM_COMMON_PROMOTED_DIMENSION
        and manifest.PHYSICAL_R32_DIMENSION + union_dimension + manifest.MEMORY_ORDER
        <= manifest.MAXIMUM_ONLINE_CONTINUOUS_DIMENSION
    )
    alignment_threshold = frozen["contract"]["common_resolved_subspace"][
        "coordinate_policy"
    ]["minimum_anchor_basis_principal_cosine_for_one_global_chart"]
    global_chart_alignment_passed = bool(np.min(anchor_cosines) >= alignment_threshold)
    arrays = {
        "common_reference_basis": reference_basis,
        "union_singular_values": union_singular,
        "primary_local_promoted_basis": primary_local["promoted_basis"],
        "heldout_local_promoted_basis": heldout_local["promoted_basis"],
        "primary_aligned_common_basis": primary_common["aligned_common_basis"],
        "heldout_aligned_common_basis": heldout_common["aligned_common_basis"],
        "anchor_common_basis_principal_cosines": anchor_cosines,
        "primary_common_stable_basis": primary_common["stable_basis"],
        "heldout_common_stable_basis": heldout_common["stable_basis"],
    }
    memory_arrays = {}
    for label, values in (("primary", primary_memory), ("heldout", heldout_memory)):
        for name, value in values.items():
            memory_arrays[f"{label}_{name}"] = value
    _write_npz(SCRATCH_DIRECTORY / "common_subspace.npz", **arrays)
    _write_npz(
        SCRATCH_DIRECTORY / "common_memory_models.npz",
        training_angular_frequencies_per_second=frequencies,
        heldout_angular_frequencies_per_second=heldout_frequencies,
        **memory_arrays,
    )
    roundtrip = True
    with np.load(SCRATCH_DIRECTORY / "common_subspace.npz", allow_pickle=False) as source:
        for name, value in arrays.items():
            roundtrip &= np.array_equal(source[name], value)
    with np.load(SCRATCH_DIRECTORY / "common_memory_models.npz", allow_pickle=False) as source:
        for name, value in memory_arrays.items():
            roundtrip &= np.array_equal(source[name], value)
    metrics = {
        "stage": "two_anchor_common_resolved_subspace_and_R96_memory",
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 1,
        "new_truth_anchors": 1,
        "common_basis_memory_fits": 2,
        "primary_local_promoted_dimension": primary_projection_metrics["promoted_dimension"],
        "heldout_local_promoted_dimension": heldout_projection_metrics["promoted_dimension"],
        "common_promoted_union_dimension": union_dimension,
        "online_continuous_dimension": manifest.PHYSICAL_R32_DIMENSION + union_dimension + manifest.MEMORY_ORDER,
        "anchor_common_basis_minimum_principal_cosine": float(np.min(anchor_cosines)),
        "anchor_common_basis_maximum_principal_angle_degrees": float(
            np.degrees(np.arccos(np.clip(np.min(anchor_cosines), -1.0, 1.0)))
        ),
        "primary_projection_metrics": primary_projection_metrics,
        "heldout_projection_metrics": heldout_projection_metrics,
        "primary_common_metrics": primary_common_metrics,
        "heldout_common_metrics": heldout_common_metrics,
        "primary_memory_metrics": primary_memory_metrics,
        "heldout_memory_metrics": heldout_memory_metrics,
        "primary_projection_passed": primary_projection_passed,
        "heldout_projection_passed": heldout_projection_passed,
        "primary_common_passed": primary_common_passed,
        "heldout_common_passed": heldout_common_passed,
        "dimension_passed": dimension_passed,
        "primary_memory_passed": primary_memory_metrics["passed"],
        "heldout_memory_passed": heldout_memory_metrics["passed"],
        "global_chart_alignment_passed": global_chart_alignment_passed,
        "database_roundtrip_bitwise": bool(roundtrip),
        "physical_failure_detected": False,
        "wall_seconds": float(time.perf_counter() - began),
    }
    metrics["numerical_passed"] = bool(
        primary_projection_passed
        and heldout_projection_passed
        and primary_common_passed
        and heldout_common_passed
        and dimension_passed
        and roundtrip
    )
    metrics["memory_passed"] = bool(
        primary_memory_metrics["passed"] and heldout_memory_metrics["passed"]
    )
    metrics["passed"] = bool(metrics["numerical_passed"] and metrics["memory_passed"])
    _write_json(SCRATCH_DIRECTORY / "common_metrics.json", metrics)
    return metrics


def _classification(assembly: dict, common: dict) -> tuple[str, str | None, bool]:
    if not assembly.get("passed", False) or not common.get("numerical_passed", False):
        return NUMERICAL_FAIL_CLASSIFICATION, None, False
    if not common.get("memory_passed", False):
        return (
            MEMORY_FAIL_CLASSIFICATION,
            "definitions_only_reduced_variable_or_memory_architecture_reassessment_manifest",
            False,
        )
    if common.get("global_chart_alignment_passed", False):
        return (
            GLOBAL_PASS_CLASSIFICATION,
            "definitions_only_R32_R96_online_prototype_manifest",
            True,
        )
    return (
        ATLAS_PASS_CLASSIFICATION,
        "definitions_only_two_chart_conservative_atlas_manifest",
        True,
    )


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": _sha(path),
                "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
            })
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
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": MANIFEST_COMMIT,
        "latest_work_package": WORK_PACKAGE,
    })
    _write_json(CANONICAL_SUMMARY, catalog)


def _finalize() -> dict:
    _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("cross-anchor preflight is already canonicalized")
    assembly = _read(SCRATCH_DIRECTORY / "assembly_metrics.json")
    common_path = SCRATCH_DIRECTORY / "common_metrics.json"
    common = _read(common_path) if common_path.exists() else {"numerical_passed": False}
    classification, authorized_next, passed = _classification(assembly, common)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "assembly_passed": assembly["passed"],
        "common_numerical_passed": common.get("numerical_passed", False),
        "memory_passed": common.get("memory_passed", False),
        "global_chart_alignment_passed": common.get("global_chart_alignment_passed", False),
        "primary_local_promoted_dimension": common.get("primary_local_promoted_dimension"),
        "heldout_local_promoted_dimension": common.get("heldout_local_promoted_dimension"),
        "common_promoted_union_dimension": common.get("common_promoted_union_dimension"),
        "online_continuous_dimension": common.get("online_continuous_dimension"),
        "anchor_common_basis_minimum_principal_cosine": common.get(
            "anchor_common_basis_minimum_principal_cosine"
        ),
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 1,
        "new_truth_anchors": 1,
        "production_memory_coefficients_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    for name in (
        "assembly_metrics.json",
        "heldout_generator.npz",
        "common_metrics.json",
        "common_subspace.npz",
        "common_memory_models.npz",
    ):
        source = SCRATCH_DIRECTORY / name
        if source.exists():
            shutil.copy2(source, CANONICAL_DIRECTORY / name)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_parent": MANIFEST_PARENT,
        "manifest_tree": MANIFEST_TREE,
        "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
        "parent_memory_package_hashes": _checksums(manifest.PARENT_DIRECTORY),
        "primary_R32_package_hashes": _checksums(manifest.R32_DIRECTORY),
    })
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py",
    )
    _write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "CERTIFIED" if passed else "REJECTED",
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean_at_start": True,
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "report": REPORT_RELATIVE,
        "source_hashes": {relative: _sha(ROOT / relative) for relative in source_files},
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": THREAD_ENVIRONMENT,
    })
    checksum_names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in checksum_names),
        encoding="utf-8",
    )
    if common_path.exists():
        primary_memory = common["primary_memory_metrics"]
        heldout_memory = common["heldout_memory_metrics"]
        detail = (
            f"Local promoted dimensions 20/16 ms and their common union are "
            f"`{common['primary_local_promoted_dimension']}/{common['heldout_local_promoted_dimension']}/"
            f"{common['common_promoted_union_dimension']}`. Online dimension is "
            f"`{common['online_continuous_dimension']}`; the minimum aligned-basis principal "
            f"cosine is `{common['anchor_common_basis_minimum_principal_cosine']:.6e}`.\n\n"
            f"Primary order-96 training/held-out max dynamic errors are "
            f"`{primary_memory['training_maximum_normalized_dynamic_transfer_relative_error']:.6e}/"
            f"{primary_memory['heldout_maximum_normalized_dynamic_transfer_relative_error']:.6e}`. "
            f"Held-out-anchor errors are "
            f"`{heldout_memory['training_maximum_normalized_dynamic_transfer_relative_error']:.6e}/"
            f"{heldout_memory['heldout_maximum_normalized_dynamic_transfer_relative_error']:.6e}`."
        )
    else:
        detail = "The held-out generator failed before common-subspace analysis."
    REPORT_PATH.write_text(
        "\n".join((
            "# Common-resolved-subspace cross-anchor preflight WP10c9d6c7c3b5c4f25o",
            "",
            "## Classification",
            "",
            f"`{classification}`",
            "",
            "One complete generator was assembled at the exact committed 16 ms state. No nonlinear root or propagation was executed.",
            "",
            detail,
            "",
            f"Authorized next artifact: `{authorized_next}`. Production coefficients, a predictive cycle, and reduced slow evolution remain blocked.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("assembly", "common", "finalize", "all"), default="all"
    )
    arguments = parser.parse_args()
    if arguments.stage in {"assembly", "all"}:
        print(json.dumps(_plain(_assembly_stage()), indent=2, sort_keys=True), flush=True)
    if arguments.stage in {"common", "all"}:
        print(json.dumps(_plain(_common_stage()), indent=2, sort_keys=True), flush=True)
    if arguments.stage in {"finalize", "all"}:
        print(json.dumps(_plain(_finalize()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
