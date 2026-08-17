#!/usr/bin/env python3
"""Run the nonpropagating single-anchor closure descriptor-schema pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
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
from scipy.linalg import null_space, schur, solve_triangular


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_pathwise_offline_closure_database_manifest_wp10c9d6c7c3b5c4f25b as manifest  # noqa: E402
import run_causal_inner_face36_state_dependent_fixed_q_step_preflight_wp10c9d6c7c3b5c4f24 as c4f24  # noqa: E402
import run_causal_inner_face36_leading_two_plus_hmm_fixed_q_preflight_wp10c9d6c7c3b5c4f22 as c4f22  # noqa: E402

from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_fixed_q_reaction,
    causal_five_field_fixed_q_reaction_jvp,
    load_causal_five_field_fixed_q_continuation_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25c"
PASS_CLASSIFICATION = (
    "single_anchor_descriptor_schema_passed_"
    "first_training_batch_manifest_authorized"
)
FAIL_CLASSIFICATION = "single_anchor_descriptor_schema_failed_database_campaign_blocked"
PARENT_PACKAGE_COMMIT = "38c70f0d57f98ecd5f6f4351235cb32f0ae748d5"
PARENT_PACKAGE_PARENT = "34bab9dad7288ab8f4484caf0d54abec2ee57e44"
PARENT_PACKAGE_TREE = "cbd0074e06cdd116a9bd69b960beb7c812a783f5"

ARTIFACT = "causal_inner_pathwise_closure_descriptor_pilot_wp10c9d6c7c3b5c4f25c"
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_pathwise_closure_descriptor_pilot_"
    "wp10c9d6c7c3b5c4f25c.py"
)
THIS_TEST = (
    "tests/test_causal_inner_pathwise_closure_descriptor_pilot_"
    "wp10c9d6c7c3b5c4f25c.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PATHWISE_CLOSURE_DESCRIPTOR_"
    "PILOT_WP10C9D6C7C3B5C4F25C_2026-08-17.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

SEED_DIRECTORY = manifest.PILOT_SEED_DIRECTORY
SEED_CHECKPOINT = SEED_DIRECTORY / "checkpoint_warm_3.npz"
AGGREGATION_SUMMARY = ROOT / (
    "results/canonical/causal_inner_face36_fixed_q_primary_evidence_"
    "aggregation_wp10c9d6c7c3b5c4f24e14r/summary.json"
)
THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
CONSERVATIVE_FIELDS = np.asarray((0, 2, 3), dtype=int)
N_FIELDS = 5
PRIMARY_CELLS = 16
RESOLVED_STORAGE_DIMENSION = PRIMARY_CELLS * N_FIELDS
EXPLICIT_MODE_DIMENSION = 2
RESOLVED_DIMENSION = RESOLVED_STORAGE_DIMENSION + EXPLICIT_MODE_DIMENSION


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
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_PACKAGE_COMMIT) != PARENT_PACKAGE_COMMIT:
        raise RuntimeError("descriptor-pilot manifest commit changed")
    if _git("rev-parse", f"{PARENT_PACKAGE_COMMIT}^") != PARENT_PACKAGE_PARENT:
        raise RuntimeError("descriptor-pilot manifest parent changed")
    if _git("rev-parse", f"{PARENT_PACKAGE_COMMIT}^{{tree}}") != PARENT_PACKAGE_TREE:
        raise RuntimeError("descriptor-pilot manifest tree changed")
    hashes = manifest._checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    pilot = _read(manifest.ARTIFACT_DIRECTORY / "pilot_contract.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "closure_database_contract.json")
    if (
        not summary["passed"]
        or not summary["single_anchor_descriptor_pilot_authorized"]
        or summary["full_anchor_campaign_authorized"]
        or summary["authorized_next"]
        != "WP10c9d6c7c3b5c4f25c_single_anchor_descriptor_pilot"
        or pilot["allowed_new_nonlinear_roots"] != 0
        or pilot["allowed_exact_continuous_descriptor_assemblies"] != 1
        or pilot["allowed_fine_layout_queries"] != 0
        or contract["identification_scope"]["online_truth_calls"] != 0
    ):
        raise RuntimeError("descriptor-pilot authorization changed")
    for relative, expected in pilot["seed_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"descriptor-pilot seed changed: {relative}")
    aggregation = _read(AGGREGATION_SUMMARY)
    if (
        not aggregation["passed"]
        or aggregation["classification"]
        != "primary_bounded_continuation_evidence_certified"
        or aggregation["accepted_primary_BDF2_roots"] != 4
    ):
        raise RuntimeError("accepted primary continuation evidence changed")
    current_threads = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if current_threads != THREAD_ENVIRONMENT:
        raise RuntimeError("descriptor-pilot thread environment is not pinned")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("descriptor pilot requires a clean tracked tree")
    return {
        "summary": summary,
        "pilot": pilot,
        "contract": contract,
        "package_hashes": hashes,
        "aggregation_summary_sha256": _sha(AGGREGATION_SUMMARY),
    }


def _coarse_groups(n_cells: int) -> tuple[tuple[int, int], ...]:
    if n_cells != 112:
        raise ValueError("descriptor pilot expects the committed 112-cell middle layout")
    groups = tuple((9 * index, 9 * (index + 1)) for index in range(8)) + tuple(
        (72 + 5 * index, 72 + 5 * (index + 1)) for index in range(8)
    )
    if groups[0][0] != 0 or groups[-1][1] != n_cells:
        raise RuntimeError("coarse groups do not cover the truth layout")
    if any(left[1] != right[0] for left, right in zip(groups[:-1], groups[1:])):
        raise RuntimeError("coarse groups are not contiguous")
    if groups[7][1] != 72 or groups[8][0] != 72:
        raise RuntimeError("coarse grid does not retain parent face 36")
    return groups


def _coarse_boundaries(n_cells: int) -> np.ndarray:
    groups = _coarse_groups(n_cells)
    return np.asarray((groups[0][0], *(group[1] for group in groups)), dtype=int)


def _incidence_matrix() -> np.ndarray:
    incidence = np.zeros((PRIMARY_CELLS, PRIMARY_CELLS + 1), dtype=float)
    for cell in range(PRIMARY_CELLS):
        incidence[cell, cell] = 1.0
        incidence[cell, cell + 1] = -1.0
    return incidence


def _continuous_generator(
    free_generator: np.ndarray,
    free_rate: np.ndarray,
    constraint: np.ndarray,
    reaction_lift: np.ndarray,
    q_row_derivatives: np.ndarray,
    lift_derivatives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimensions = int(free_rate.size)
    constraint_count = int(constraint.shape[0])
    if free_generator.shape != (dimensions, dimensions):
        raise ValueError("free generator must be square")
    multiplier = -constraint @ free_rate
    fixed_rate = free_rate + reaction_lift @ multiplier
    complete = np.empty_like(free_generator)
    multiplier_jacobian = np.empty((constraint_count, dimensions), dtype=float)
    for column in range(dimensions):
        d_constraint = q_row_derivatives[column]
        d_lift = lift_derivatives[column]
        d_multiplier = -(
            d_constraint @ free_rate + constraint @ free_generator[:, column]
        )
        multiplier_jacobian[:, column] = d_multiplier
        complete[:, column] = (
            free_generator[:, column]
            + d_lift @ multiplier
            + reaction_lift @ d_multiplier
        )
    return complete, fixed_rate, multiplier_jacobian


def _continuous_generator_action(
    free_action: np.ndarray,
    free_rate: np.ndarray,
    constraint: np.ndarray,
    reaction_lift: np.ndarray,
    q_row_derivative: np.ndarray,
    lift_derivative: np.ndarray,
) -> np.ndarray:
    """Apply the complete constrained-rate derivative in one direction."""

    multiplier = -constraint @ free_rate
    d_multiplier = -(q_row_derivative @ free_rate + constraint @ free_action)
    return free_action + lift_derivative @ multiplier + reaction_lift @ d_multiplier


def _coarse_storage_restriction(
    descriptor: np.ndarray, conservation_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(conservation_rows, dtype=float)
    n_cells = int(rows.shape[0])
    groups = _coarse_groups(n_cells)
    physical_descriptor = C * rows.ravel()[:, None] * descriptor
    unscaled = np.zeros((RESOLVED_STORAGE_DIMENSION, descriptor.shape[1]))
    for coarse_cell, (start, stop) in enumerate(groups):
        for field in range(N_FIELDS):
            target = N_FIELDS * coarse_cell + field
            source_rows = N_FIELDS * np.arange(start, stop) + field
            unscaled[target] = np.sum(physical_descriptor[source_rows], axis=0)
    row_scales = np.linalg.norm(unscaled, axis=1)
    if np.any(~np.isfinite(row_scales)) or np.any(row_scales <= 0.0):
        raise RuntimeError("coarse storage restriction lost a row")
    return unscaled / row_scales[:, None], unscaled, row_scales


def _resolved_projection(
    restriction: np.ndarray, a2_dual: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    resolved = np.vstack((restriction, a2_dual))
    gram = resolved @ resolved.T
    lifting = resolved.T @ np.linalg.solve(gram, np.eye(resolved.shape[0]))
    identity_defect = float(np.max(np.abs(resolved @ lifting - np.eye(resolved.shape[0]))))
    complement = null_space(resolved)
    metrics = {
        "resolved_rank": int(np.linalg.matrix_rank(resolved)),
        "resolved_dimension": int(resolved.shape[0]),
        "unresolved_dimension": int(complement.shape[1]),
        "restriction_lifting_identity_defect": identity_defect,
        "resolved_condition_number": float(np.linalg.cond(resolved)),
        "complement_orthogonality_defect": float(
            np.max(np.abs(complement.T @ complement - np.eye(complement.shape[1])))
        ),
        "lifting_complement_defect": float(np.max(np.abs(complement.T @ lifting))),
    }
    return resolved, lifting, complement, metrics


def _transfer_from_schur(
    generator: np.ndarray,
    lifting: np.ndarray,
    complement: np.ndarray,
    output_map: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    unresolved = complement.T @ generator @ complement
    forcing = complement.T @ generator @ lifting
    observation = output_map @ complement
    direct = output_map @ lifting
    triangular, vectors = schur(unresolved, output="complex")
    transformed_forcing = vectors.conj().T @ forcing
    identity = np.eye(triangular.shape[0], dtype=complex)
    values = []
    maximum_residual = 0.0
    conjugacy_defects = []
    frequencies_with_dc = np.concatenate(([0.0], np.asarray(frequencies, dtype=float)))
    conjugacy_indices = {1, 1 + len(frequencies) // 2, len(frequencies)}
    for slot, omega in enumerate(frequencies_with_dc):
        system = 1j * omega * identity - triangular
        transformed = solve_triangular(system, transformed_forcing, lower=False)
        solved = vectors @ transformed
        transfer = direct + observation @ solved
        values.append(transfer)
        original_system = 1j * omega * np.eye(unresolved.shape[0]) - unresolved
        residual = original_system @ solved - forcing
        scale = max(
            float(np.linalg.norm(original_system @ solved)),
            float(np.linalg.norm(forcing)),
            np.finfo(float).tiny,
        )
        maximum_residual = max(maximum_residual, float(np.linalg.norm(residual) / scale))
        if slot in conjugacy_indices:
            negative_system = -1j * omega * identity - triangular
            negative = vectors @ solve_triangular(
                negative_system, transformed_forcing, lower=False
            )
            negative_transfer = direct + observation @ negative
            conjugacy_defects.append(_relative(negative_transfer, np.conjugate(transfer)))
    transfers = np.asarray(values)
    reconstruction = vectors @ triangular @ vectors.conj().T
    metrics = {
        "frequency_count_including_DC": int(frequencies_with_dc.size),
        "maximum_frequency_solve_relative_residual": maximum_residual,
        "maximum_transfer_conjugate_symmetry_relative_defect": max(
            conjugacy_defects, default=0.0
        ),
        "complex_schur_reconstruction_relative_defect": _relative(
            reconstruction, unresolved
        ),
        "complex_schur_unitarity_defect": float(
            np.max(np.abs(vectors.conj().T @ vectors - np.eye(vectors.shape[1])))
        ),
        "unstable_unresolved_pole_count_diagnostic": int(
            np.count_nonzero(np.real(np.diag(triangular)) >= 0.0)
        ),
        "maximum_transfer_absolute_value": float(np.max(np.abs(transfers))),
    }
    return transfers, np.diag(triangular), metrics


def _seed_data() -> dict:
    layout, configuration, _trajectory, *_unused = c4f24._endpoint_data()
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(112, N_FIELDS)
    rows = np.asarray(configuration["rows"], dtype=float).reshape(112, N_FIELDS)
    continuation = load_causal_five_field_fixed_q_continuation_state(
        SEED_CHECKPOINT, context
    )
    state = np.asarray(continuation.current_primitive_charts, dtype=float)
    if (
        continuation.current_order != 2
        or continuation.next_order != 2
        or continuation.completed_steps < 4
        or continuation.nonlinear_solver_state is None
        or not np.array_equal(
            continuation.nonlinear_solver_state.anchor_primitive_charts, state
        )
    ):
        raise RuntimeError("accepted descriptor-pilot seed semantics changed")
    return {
        "layout": layout,
        "context": context,
        "columns": columns,
        "rows": rows,
        "state": state,
        "continuation": continuation,
    }


def _assembly_stage() -> dict:
    if SCRATCH_DIRECTORY.exists():
        raise RuntimeError("descriptor-pilot scratch directory already exists")
    SCRATCH_DIRECTORY.mkdir(parents=True)
    began = time.perf_counter()
    data = _seed_data()
    state = data["state"]
    context = data["context"]
    columns = data["columns"]
    rows = data["rows"]
    layout = data["layout"]
    reaction = causal_five_field_fixed_q_reaction(
        context,
        state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
        maximum_schur_condition_number=1.0e8,
    )
    tangent = causal_five_field_monolithic_frozen_tangent(
        context,
        state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    dimensions = int(state.size)
    full_jvp = causal_five_field_fixed_q_reaction_jvp(
        context,
        state,
        np.eye(dimensions),
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
        reaction=reaction,
    )
    complete, fixed_rate, multiplier_jacobian = _continuous_generator(
        tangent.scaled_generator_per_s,
        tangent.scaled_base_rate_per_s,
        reaction.q3_scaled_derivative,
        reaction.reaction_lift,
        full_jvp.q3_scaled_row_derivatives,
        full_jvp.reaction_lift_derivatives,
    )
    saved = np.asarray(c4f22._saved_directions("middle")[:2], dtype=float)
    leading_scaled = saved.reshape(2, -1).T / columns.ravel()[:, None]
    leading_fixed, _ = c4f22._physical_reaction_projection(
        leading_scaled,
        reaction.q3_scaled_derivative,
        reaction.reaction_lift,
    )
    duals = c4f22._stable_a2_duals(
        reaction.descriptor_scaled_matrix,
        reaction.q3_scaled_derivative,
        reaction.reaction_lift,
        leading_fixed,
    )
    restriction, physical_restriction, restriction_scales = (
        _coarse_storage_restriction(reaction.descriptor_scaled_matrix, rows)
    )
    resolved, lifting, complement, projection_metrics = _resolved_projection(
        restriction, duals["dual_qr"]
    )
    constraint_reconstruction = (
        reaction.q3_scaled_derivative @ lifting @ resolved
    )
    constraint_rowspace_defect = _relative(
        constraint_reconstruction, reaction.q3_scaled_derivative
    )
    boundaries = _coarse_boundaries(state.shape[0])
    shared = np.asarray(
        tangent.spatial_tangent.shared_face_flux_scaled_jacobians,
        dtype=float,
    )
    output_map = shared[boundaries][:, CONSERVATIVE_FIELDS].reshape(-1, dimensions)
    incidence = _incidence_matrix()
    telescope = np.sum(incidence, axis=0)
    expected_telescope = np.zeros(PRIMARY_CELLS + 1)
    expected_telescope[0] = 1.0
    expected_telescope[-1] = -1.0
    telescope_defect = float(np.max(np.abs(telescope - expected_telescope)))
    audit_direction = leading_fixed[:, 0] / np.linalg.norm(leading_fixed[:, 0])
    directional_jvp = causal_five_field_fixed_q_reaction_jvp(
        context,
        state,
        audit_direction,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
        reaction=reaction,
    )
    direct_vector = _continuous_generator_action(
        tangent.scaled_generator_per_s @ audit_direction,
        tangent.scaled_base_rate_per_s,
        reaction.q3_scaled_derivative,
        reaction.reaction_lift,
        directional_jvp.q3_scaled_row_derivatives[0],
        directional_jvp.reaction_lift_derivatives[0],
    )
    assembled_vector = complete @ audit_direction
    jvp_defect = _relative(direct_vector, assembled_vector)
    constraint_differential = np.empty((3, dimensions))
    for column in range(dimensions):
        constraint_differential[:, column] = (
            reaction.q3_scaled_derivative @ complete[:, column]
            + full_jvp.q3_scaled_row_derivatives[column] @ fixed_rate
        )
    differential_scale = max(float(np.linalg.norm(complete)), 1.0)
    constraint_differential_defect = float(
        np.linalg.norm(constraint_differential) / differential_scale
    )
    descriptor = np.asarray(reaction.descriptor_scaled_matrix, dtype=float)
    stationary = -descriptor @ complete
    descriptor_closure = _relative(descriptor @ complete + stationary, np.zeros_like(stationary))
    gates = _read(manifest.ARTIFACT_DIRECTORY / "pilot_contract.json")["pass_requires"]
    metrics = {
        "seed_checkpoint_sha256": _sha(SEED_CHECKPOINT),
        "seed_completed_steps": data["continuation"].completed_steps,
        "seed_elapsed_time_seconds": data["continuation"].elapsed_time_seconds,
        "seed_branch_label": "unclassified",
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "exact_continuous_descriptor_assemblies": 1,
        "truth_state_dimension": dimensions,
        "coarse_storage_dimension": RESOLVED_STORAGE_DIMENSION,
        "explicit_mode_dimension": EXPLICIT_MODE_DIMENSION,
        "resolved_dimension": RESOLVED_DIMENSION,
        "output_dimension": int(output_map.shape[0]),
        "coarse_face_indices": boundaries,
        "parent_face36_truth_index": 72,
        "parent_face36_coarse_index": int(np.flatnonzero(boundaries == 72)[0]),
        "complete_JVP_relative_defect": jvp_defect,
        "constraint_differential_identity_relative_defect": constraint_differential_defect,
        "descriptor_generator_closure_relative_defect": descriptor_closure,
        "reaction_identity_directional_defect": full_jvp.maximum_identity_directional_defect,
        "reaction_ledger_directional_relative_defect": (
            full_jvp.maximum_reaction_ledger_directional_relative_defect
        ),
        "constraint_rowspace_relative_defect": constraint_rowspace_defect,
        "M_J_E_telescope_relative_defect": telescope_defect,
        "a2_dual_biorthogonality_defect": duals["qr_metrics"]["biorthogonality_defect"],
        "a2_dual_reaction_annihilation_defect": duals["qr_metrics"][
            "normalized_reaction_annihilation_defect"
        ],
        **projection_metrics,
        "wall_seconds": float(time.perf_counter() - began),
    }
    metrics["passed"] = bool(
        metrics["new_nonlinear_roots"] == 0
        and metrics["propagated_states"] == 0
        and metrics["exact_continuous_descriptor_assemblies"] == 1
        and metrics["complete_JVP_relative_defect"]
        <= gates["descriptor_complete_JVP_relative_defect_max"]
        and metrics["restriction_lifting_identity_defect"]
        <= gates["restriction_lifting_identity_max"]
        and metrics["M_J_E_telescope_relative_defect"]
        <= gates["M_J_E_telescope_relative_defect_max"]
        and metrics["resolved_rank"] == RESOLVED_DIMENSION
        and metrics["constraint_rowspace_relative_defect"] <= 5.0e-10
        and metrics["constraint_differential_identity_relative_defect"] <= 5.0e-10
    )
    _write_npz(SCRATCH_DIRECTORY / "descriptor_E.npz", descriptor=descriptor)
    _write_npz(
        SCRATCH_DIRECTORY / "descriptor_A.npz",
        complete_fixed_Q_generator=complete,
        fixed_Q_rate=fixed_rate,
        multiplier_jacobian=multiplier_jacobian,
    )
    _write_npz(
        SCRATCH_DIRECTORY / "projection.npz",
        normalized_storage_restriction=restriction,
        physical_storage_restriction=physical_restriction,
        storage_restriction_row_scales=restriction_scales,
        resolved_restriction=resolved,
        resolved_lifting=lifting,
        unresolved_orthonormal_basis=complement,
        leading_fixed_Q_lifts=leading_fixed,
        a2_dual=duals["dual_qr"],
        output_map=output_map,
        coarse_face_indices=boundaries,
        incidence_matrix=incidence,
    )
    _write_json(SCRATCH_DIRECTORY / "assembly_metrics.json", metrics)
    return metrics


def _transfer_stage() -> dict:
    assembly = _read(SCRATCH_DIRECTORY / "assembly_metrics.json")
    if not assembly["passed"]:
        raise RuntimeError("binding assembly gate failed; transfer stage is blocked")
    began = time.perf_counter()
    with np.load(SCRATCH_DIRECTORY / "descriptor_A.npz", allow_pickle=False) as source:
        generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    with np.load(SCRATCH_DIRECTORY / "projection.npz", allow_pickle=False) as source:
        lifting = np.asarray(source["resolved_lifting"], dtype=float)
        complement = np.asarray(source["unresolved_orthonormal_basis"], dtype=float)
        output_map = np.asarray(source["output_map"], dtype=float)
    frequencies = np.asarray(manifest._frequency_grid()["values_per_second"], dtype=float)
    transfer, poles, metrics = _transfer_from_schur(
        generator, lifting, complement, output_map, frequencies
    )
    gates = _read(manifest.ARTIFACT_DIRECTORY / "pilot_contract.json")["pass_requires"]
    _write_npz(
        SCRATCH_DIRECTORY / "transfer_real.npz",
        angular_frequencies_per_second=np.concatenate(([0.0], frequencies)),
        transfer_real=np.real(transfer),
        unresolved_pole_real=np.real(poles),
    )
    _write_npz(
        SCRATCH_DIRECTORY / "transfer_imag.npz",
        angular_frequencies_per_second=np.concatenate(([0.0], frequencies)),
        transfer_imag=np.imag(transfer),
        unresolved_pole_imag=np.imag(poles),
    )
    roundtrip = True
    with np.load(SCRATCH_DIRECTORY / "transfer_real.npz", allow_pickle=False) as real_source:
        roundtrip &= np.array_equal(real_source["transfer_real"], np.real(transfer))
    with np.load(SCRATCH_DIRECTORY / "transfer_imag.npz", allow_pickle=False) as imag_source:
        roundtrip &= np.array_equal(imag_source["transfer_imag"], np.imag(transfer))
    metrics.update({
        "database_roundtrip_bitwise": bool(roundtrip),
        "wall_seconds": float(time.perf_counter() - began),
    })
    metrics["passed"] = bool(
        metrics["frequency_count_including_DC"] == manifest.FREQUENCY_COUNT + 1
        and metrics["maximum_frequency_solve_relative_residual"]
        <= gates["frequency_solve_relative_residual_max"]
        and metrics["maximum_transfer_conjugate_symmetry_relative_defect"]
        <= gates["transfer_conjugate_symmetry_relative_defect_max"]
        and metrics["database_roundtrip_bitwise"]
    )
    _write_json(SCRATCH_DIRECTORY / "transfer_metrics.json", metrics)
    return metrics


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
        "latest_source_parent_commit": PARENT_PACKAGE_COMMIT,
        "latest_work_package": WORK_PACKAGE,
    })
    _write_json(CANONICAL_SUMMARY, catalog)


def _finalize() -> dict:
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("descriptor pilot is already canonicalized")
    assembly = _read(SCRATCH_DIRECTORY / "assembly_metrics.json")
    transfer_path = SCRATCH_DIRECTORY / "transfer_metrics.json"
    transfer = _read(transfer_path) if transfer_path.exists() else {"passed": False}
    passed = bool(assembly["passed"] and transfer["passed"])
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "pilot_executed": True,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "exact_continuous_descriptor_assemblies": 1,
        "seed_branch_label": "unclassified",
        "truth_state_dimension": assembly["truth_state_dimension"],
        "resolved_dimension": assembly["resolved_dimension"],
        "unresolved_dimension": assembly["unresolved_dimension"],
        "output_dimension": assembly["output_dimension"],
        "assembly_passed": assembly["passed"],
        "transfer_passed": transfer["passed"],
        "first_training_batch_manifest_authorized": passed,
        "full_anchor_campaign_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": (
            "definitions_only_first_training_anchor_batch_manifest" if passed else None
        ),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True)
    names = (
        "assembly_metrics.json",
        "descriptor_A.npz",
        "descriptor_E.npz",
        "projection.npz",
        "transfer_metrics.json",
        "transfer_real.npz",
        "transfer_imag.npz",
    )
    for name in names:
        source = SCRATCH_DIRECTORY / name
        if source.exists():
            shutil.copy2(source, CANONICAL_DIRECTORY / name)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "seed_lock.json", {
        "seed_checkpoint": str(SEED_CHECKPOINT.relative_to(ROOT)),
        "seed_checkpoint_sha256": _sha(SEED_CHECKPOINT),
        "aggregation_summary": str(AGGREGATION_SUMMARY.relative_to(ROOT)),
        "aggregation_summary_sha256": _sha(AGGREGATION_SUMMARY),
        "manifest_commit": PARENT_PACKAGE_COMMIT,
        "manifest_tree": PARENT_PACKAGE_TREE,
    })
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py",
        "scripts/run_causal_inner_face36_state_dependent_fixed_q_step_preflight_"
        "wp10c9d6c7c3b5c4f24.py",
        "scripts/run_causal_inner_face36_leading_two_plus_hmm_fixed_q_preflight_"
        "wp10c9d6c7c3b5c4f22.py",
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
    checksum_names = tuple(
        sorted(path.name for path in CANONICAL_DIRECTORY.iterdir() if path.is_file())
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in checksum_names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join((
            "# Pathwise closure descriptor pilot WP10c9d6c7c3b5c4f25c",
            "",
            "## Classification",
            "",
            f"`{classification}`",
            "",
            "The hash-locked accepted primary 20 ms continuation checkpoint was used only as an unclassified schema seed. No nonlinear root was solved, no state was propagated, and no trajectory time was added.",
            "",
            f"The exact continuous fixed-Q rate derivative has dimension `{assembly['truth_state_dimension']}`. Conservative restriction plus two explicit stable-mode duals produce `{assembly['resolved_dimension']}` resolved coordinates and `{assembly['unresolved_dimension']}` orthogonal unresolved coordinates. The all-face M/J/E output has dimension `{assembly['output_dimension']}`.",
            "",
            f"Assembly pass: `{assembly['passed']}`. Complete JVP defect: `{assembly['complete_JVP_relative_defect']:.6e}`. Restriction/lifting identity defect: `{assembly['restriction_lifting_identity_defect']:.6e}`. Constraint-rowspace defect: `{assembly['constraint_rowspace_relative_defect']:.6e}`. M/J/E telescope defect: `{assembly['M_J_E_telescope_relative_defect']:.6e}`.",
            "",
            f"Transfer pass: `{transfer.get('passed', False)}`. Maximum frequency solve residual: `{transfer.get('maximum_frequency_solve_relative_residual', float('nan')):.6e}`. Conjugate-symmetry defect: `{transfer.get('maximum_transfer_conjugate_symmetry_relative_defect', float('nan')):.6e}`. Unstable unresolved poles are diagnostic here and total `{transfer.get('unstable_unresolved_pole_count_diagnostic', -1)}`; stability becomes binding only for the fitted finite-memory model.",
            "",
            "A pass authorizes only a definitions-only first training-anchor batch manifest. The seed remains unclassified, the full database campaign remains blocked, and no online solver or predictive cycle is authorized.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("assembly", "transfer", "finalize", "all"), default="all"
    )
    arguments = parser.parse_args()
    _validate_manifest(require_clean=True)
    if arguments.stage in {"assembly", "all"}:
        print(json.dumps(_plain(_assembly_stage()), indent=2, sort_keys=True), flush=True)
    if arguments.stage in {"transfer", "all"}:
        print(json.dumps(_plain(_transfer_stage()), indent=2, sort_keys=True), flush=True)
    if arguments.stage in {"finalize", "all"}:
        print(json.dumps(_plain(_finalize()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
