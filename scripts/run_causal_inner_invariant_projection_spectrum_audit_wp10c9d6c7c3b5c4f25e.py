#!/usr/bin/env python3
"""Execute the invariant-compatible projection and saved-generator spectrum audit."""

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
from scipy.linalg import qr, schur, solve_triangular


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_invariant_projection_spectrum_manifest_wp10c9d6c7c3b5c4f25d as manifest  # noqa: E402
import run_causal_inner_pathwise_closure_descriptor_pilot_wp10c9d6c7c3b5c4f25c as parent  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    _q3_physical_selectors,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    _descriptor_matrices,
    _node_reconstruction_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e"
MANIFEST_COMMIT = "d233d967141d8d63f1a595cbd518186a6084899f"
MANIFEST_PARENT = "928892fceba48a0e46d21e7e9bc2c8f41e46afa7"
MANIFEST_TREE = "ee3d99027e7dcae7775b58f1fec130f899c2c1ae"

ARTIFACT = "causal_inner_invariant_projection_spectrum_audit_wp10c9d6c7c3b5c4f25e"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_invariant_projection_spectrum_audit_"
    "wp10c9d6c7c3b5c4f25e.py"
)
THIS_TEST = (
    "tests/test_causal_inner_invariant_projection_spectrum_audit_"
    "wp10c9d6c7c3b5c4f25e.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_INVARIANT_PROJECTION_SPECTRUM_"
    "AUDIT_WP10C9D6C7C3B5C4F25E_2026-08-17.md"
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
THERMAL_STRESS_FIELDS = (1, 4)

PASS_CLASSIFICATION = (
    "invariant_projection_spectrum_transfer_passed_mode_selection_manifest_authorized"
)
UNSTABLE_CLASSIFICATION = (
    "invariant_projection_transfer_passed_unstable_modes_require_promotion"
)
CYCLE_SCALE_CLASSIFICATION = (
    "invariant_projection_transfer_passed_cycle_scale_modes_require_promotion"
)
STAGE_1_FAIL_CLASSIFICATION = "invariant_projection_failed_spectrum_blocked"
STAGE_2_FAIL_CLASSIFICATION = "invariant_projection_passed_spectrum_transfer_failed"


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


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


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
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    scale = max(
        float(np.linalg.norm(left_array)),
        float(np.linalg.norm(right_array)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left_array - right_array) / scale)


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("projection/spectrum manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("projection/spectrum manifest parent changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("projection/spectrum manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["stage_1_projection_authorized"]
        or not summary["stage_2_spectrum_transfer_authorized_only_after_stage_1_pass"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["execution_budget"]["allowed_new_nonlinear_roots"] != 0
        or contract["execution_budget"]["allowed_new_full_560_direction_descriptor_assemblies"] != 0
    ):
        raise RuntimeError("projection/spectrum authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(parent.CANONICAL_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent decisive array changed: {name}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("projection/spectrum execution requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _candidate_storage_restriction(
    mapped: np.ndarray,
    height: np.ndarray,
    conservation_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapped_array = np.asarray(mapped, dtype=float)
    height_array = np.asarray(height, dtype=float)
    rows = np.asarray(conservation_rows, dtype=float)
    if mapped_array.shape != height_array.shape or mapped_array.shape[0] != rows.size:
        raise ValueError("mapped/height descriptor dimensions changed")
    complete = mapped_array + height_array
    physical_mapped = C * rows.ravel()[:, None] * mapped_array
    physical_complete = C * rows.ravel()[:, None] * complete
    groups = parent._coarse_groups(rows.shape[0])
    unscaled = np.zeros((manifest.STORAGE_DIMENSION, mapped_array.shape[1]))
    for coarse_cell, (start, stop) in enumerate(groups):
        for field in range(manifest.FIELDS_PER_CELL):
            target = manifest.FIELDS_PER_CELL * coarse_cell + field
            source_rows = manifest.FIELDS_PER_CELL * np.arange(start, stop) + field
            source = physical_mapped if field in CONSERVATIVE_FIELDS else physical_complete
            unscaled[target] = np.sum(source[source_rows], axis=0)
    row_scales = np.linalg.norm(unscaled, axis=1)
    if np.any(~np.isfinite(row_scales)) or np.any(row_scales <= 0.0):
        raise RuntimeError("invariant-compatible storage restriction lost a row")
    return unscaled / row_scales[:, None], unscaled, row_scales


def _complete_qr_projection(
    storage_restriction: np.ndarray,
    stable_dual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    resolved = np.vstack(
        (np.asarray(storage_restriction, dtype=float), np.asarray(stable_dual, dtype=float))
    )
    orthogonal, triangular_full = qr(resolved.T, mode="full", pivoting=False)
    dimension = resolved.shape[0]
    upper = np.asarray(triangular_full[:dimension], dtype=float)
    if np.linalg.matrix_rank(upper) != dimension:
        raise RuntimeError("resolved restriction is rank deficient")
    inverse_transpose = solve_triangular(
        upper.T, np.eye(dimension), lower=True, check_finite=True
    )
    lifting = orthogonal[:, :dimension] @ inverse_transpose
    complement = orthogonal[:, dimension:]
    metrics = {
        "resolved_rank": int(np.linalg.matrix_rank(resolved)),
        "resolved_dimension": int(dimension),
        "unresolved_dimension": int(complement.shape[1]),
        "resolved_condition_number": float(np.linalg.cond(resolved)),
        "restriction_lifting_identity_defect": float(
            np.max(np.abs(resolved @ lifting - np.eye(dimension)))
        ),
        "restriction_complement_annihilation_defect": float(
            np.max(np.abs(resolved @ complement))
        ),
        "complement_orthogonality_defect": float(
            np.max(
                np.abs(complement.T @ complement - np.eye(complement.shape[1]))
            )
        ),
        "lifting_complement_defect": float(np.max(np.abs(complement.T @ lifting))),
    }
    return resolved, lifting, complement, metrics


def _stage_1() -> dict:
    frozen = _validate_manifest(require_clean=True)
    began = time.perf_counter()
    data = parent._seed_data()
    state = np.asarray(data["state"], dtype=float)
    context = data["context"]
    columns = np.asarray(data["columns"], dtype=float)
    rows = np.asarray(data["rows"], dtype=float)
    (
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        reconstruction_defect,
        partition_defect,
    ) = _node_reconstruction_weights(context, state)
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
    with np.load(parent.CANONICAL_DIRECTORY / "descriptor_E.npz", allow_pickle=False) as source:
        saved_descriptor = np.asarray(source["descriptor"], dtype=float)
    descriptor_parity = _relative(mapped + height, saved_descriptor)
    with np.load(parent.CANONICAL_DIRECTORY / "projection.npz", allow_pickle=False) as source:
        stable_dual = np.asarray(source["a2_dual"], dtype=float)
        output_map = np.asarray(source["output_map"], dtype=float)
        coarse_face_indices = np.asarray(source["coarse_face_indices"], dtype=int)
        incidence = np.asarray(source["incidence_matrix"], dtype=float)
    storage, physical_storage, storage_scales = _candidate_storage_restriction(
        mapped, height, rows
    )
    resolved, lifting, complement, projection_metrics = _complete_qr_projection(
        storage, stable_dual
    )
    q_selectors = _q3_physical_selectors(state.shape[0], 72, rows)
    q_physical = np.asarray(q_selectors @ mapped, dtype=float)
    q_norms = np.linalg.norm(q_physical, axis=1)
    q_scaled = q_physical / q_norms[:, None]
    constraint_defect = _relative(q_scaled @ lifting @ resolved, q_scaled)
    physical_mapped = C * rows.ravel()[:, None] * mapped
    truth_totals = []
    coarse_totals = []
    for field in CONSERVATIVE_FIELDS:
        truth_totals.append(np.sum(physical_mapped[field::manifest.FIELDS_PER_CELL], axis=0))
        coarse_totals.append(
            np.sum(physical_storage[field::manifest.FIELDS_PER_CELL], axis=0)
        )
    telescope_defect = _relative(np.asarray(coarse_totals), np.asarray(truth_totals))
    expected_incidence = np.zeros(manifest.PRIMARY_CELLS + 1)
    expected_incidence[0] = 1.0
    expected_incidence[-1] = -1.0
    incidence_defect = float(np.max(np.abs(np.sum(incidence, axis=0) - expected_incidence)))
    parent_metrics = _read(parent.CANONICAL_DIRECTORY / "assembly_metrics.json")
    gates = frozen["contract"]["stage_1_projection"]["pass_requires"]
    metrics = {
        "stage": "invariant_compatible_projection",
        "candidate_id": frozen["contract"]["stage_1_projection"]["candidate_id"],
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_descriptor_assemblies": 0,
        "seed_local_mapped_height_descriptor_reconstructions": 1,
        "truth_dimension": int(state.size),
        "storage_dimension": int(storage.shape[0]),
        "explicit_mode_dimension": int(stable_dual.shape[0]),
        "constraint_dimension": int(q_scaled.shape[0]),
        "saved_complete_descriptor_relative_parity_defect": descriptor_parity,
        "constraint_rowspace_relative_defect": constraint_defect,
        "M_J_E_telescope_relative_defect": telescope_defect,
        "incidence_telescope_absolute_defect": incidence_defect,
        "node_reconstruction_relative_defect": reconstruction_defect,
        "node_partition_of_unity_defect": partition_defect,
        "a2_dual_biorthogonality_defect": parent_metrics[
            "a2_dual_biorthogonality_defect"
        ],
        "a2_dual_reaction_annihilation_defect": parent_metrics[
            "a2_dual_reaction_annihilation_defect"
        ],
        "parent_failed_constraint_rowspace_relative_defect": parent_metrics[
            "constraint_rowspace_relative_defect"
        ],
        **projection_metrics,
        "wall_seconds": float(time.perf_counter() - began),
    }
    metrics["passed"] = bool(
        metrics["new_nonlinear_roots"] == 0
        and metrics["propagated_states"] == 0
        and metrics["new_full_560_direction_descriptor_assemblies"] == 0
        and metrics["seed_local_mapped_height_descriptor_reconstructions"] == 1
        and metrics["truth_dimension"] == manifest.TRUTH_DIMENSION
        and metrics["resolved_rank"] == gates["resolved_rank"]
        and metrics["resolved_dimension"] == manifest.RESOLVED_DIMENSION
        and metrics["resolved_condition_number"] <= gates["resolved_condition_number_max"]
        and metrics["restriction_lifting_identity_defect"]
        <= gates["restriction_lifting_identity_max"]
        and metrics["restriction_complement_annihilation_defect"]
        <= gates["restriction_complement_annihilation_max"]
        and metrics["complement_orthogonality_defect"]
        <= gates["complement_orthogonality_max"]
        and metrics["constraint_rowspace_relative_defect"]
        <= gates["constraint_rowspace_relative_defect_max"]
        and metrics["saved_complete_descriptor_relative_parity_defect"]
        <= gates["saved_complete_descriptor_relative_parity_max"]
        and metrics["M_J_E_telescope_relative_defect"]
        <= gates["M_J_E_telescope_relative_defect_max"]
        and metrics["incidence_telescope_absolute_defect"]
        <= gates["M_J_E_telescope_relative_defect_max"]
        and metrics["a2_dual_biorthogonality_defect"]
        <= gates["a2_dual_biorthogonality_defect_max"]
        and metrics["a2_dual_reaction_annihilation_defect"]
        <= gates["a2_dual_reaction_annihilation_defect_max"]
    )
    _write_npz(
        SCRATCH_DIRECTORY / "projection.npz",
        normalized_storage_restriction=storage,
        physical_storage_restriction=physical_storage,
        storage_restriction_row_scales=storage_scales,
        fixed_Q_scaled_rows=q_scaled,
        fixed_Q_physical_rows=q_physical,
        fixed_Q_row_norms=q_norms,
        resolved_restriction=resolved,
        resolved_lifting=lifting,
        unresolved_orthonormal_basis=complement,
        a2_dual=stable_dual,
        output_map=output_map,
        coarse_face_indices=coarse_face_indices,
        incidence_matrix=incidence,
    )
    _write_json(SCRATCH_DIRECTORY / "stage_1_metrics.json", metrics)
    return metrics


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
            negative = vectors @ solve_triangular(
                -1j * omega * identity - triangular,
                transformed_forcing,
                lower=False,
            )
            negative_transfer = direct + observation @ negative
            conjugacy_defects.append(_relative(negative_transfer, np.conjugate(transfer)))
    transfers = np.asarray(values)
    poles = np.diag(triangular)
    reconstruction = vectors @ triangular @ vectors.conj().T
    real_parts = np.real(poles)
    omega_min = float(np.min(frequencies))
    unstable = real_parts >= 0.0
    cycle_scale = (real_parts < 0.0) & (-real_parts <= omega_min)
    stable_real = real_parts[real_parts < 0.0]
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
        "unresolved_dimension": int(unresolved.shape[0]),
        "unstable_unresolved_pole_count": int(np.count_nonzero(unstable)),
        "cycle_scale_stable_unresolved_pole_count": int(np.count_nonzero(cycle_scale)),
        "spectral_abscissa_per_second": float(np.max(real_parts)),
        "minimum_real_part_per_second": float(np.min(real_parts)),
        "slowest_stable_decay_time_seconds": (
            float(np.max(-1.0 / stable_real)) if stable_real.size else math.inf
        ),
        "fastest_stable_decay_time_seconds": (
            float(np.min(-1.0 / stable_real)) if stable_real.size else math.inf
        ),
        "complex_pole_count": int(np.count_nonzero(np.abs(np.imag(poles)) > 0.0)),
        "maximum_transfer_absolute_value": float(np.max(np.abs(transfers))),
    }
    return transfers, poles, metrics


def _stage_2() -> dict:
    frozen = _validate_manifest(require_clean=True)
    stage_1 = _read(SCRATCH_DIRECTORY / "stage_1_metrics.json")
    if not stage_1["passed"]:
        raise RuntimeError("Stage 1 projection failed; Stage 2 is blocked")
    began = time.perf_counter()
    with np.load(parent.CANONICAL_DIRECTORY / "descriptor_A.npz", allow_pickle=False) as source:
        generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    with np.load(SCRATCH_DIRECTORY / "projection.npz", allow_pickle=False) as source:
        lifting = np.asarray(source["resolved_lifting"], dtype=float)
        complement = np.asarray(source["unresolved_orthonormal_basis"], dtype=float)
        output_map = np.asarray(source["output_map"], dtype=float)
    frequencies = np.asarray(
        frozen["contract"]["stage_2_spectrum_transfer"]["frequency_grid"][
            "values_per_second"
        ],
        dtype=float,
    )
    transfer, poles, metrics = _transfer_from_schur(
        generator, lifting, complement, output_map, frequencies
    )
    _write_npz(
        SCRATCH_DIRECTORY / "transfer_real.npz",
        angular_frequencies_per_second=np.concatenate(([0.0], frequencies)),
        transfer_real=np.real(transfer),
    )
    _write_npz(
        SCRATCH_DIRECTORY / "transfer_imag.npz",
        angular_frequencies_per_second=np.concatenate(([0.0], frequencies)),
        transfer_imag=np.imag(transfer),
    )
    _write_npz(
        SCRATCH_DIRECTORY / "unresolved_poles.npz",
        pole_real_per_second=np.real(poles),
        pole_imag_per_second=np.imag(poles),
    )
    roundtrip = True
    with np.load(SCRATCH_DIRECTORY / "transfer_real.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["transfer_real"], np.real(transfer))
    with np.load(SCRATCH_DIRECTORY / "transfer_imag.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["transfer_imag"], np.imag(transfer))
    with np.load(SCRATCH_DIRECTORY / "unresolved_poles.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["pole_real_per_second"], np.real(poles))
        roundtrip &= np.array_equal(source["pole_imag_per_second"], np.imag(poles))
    metrics.update({
        "stage": "saved_generator_spectrum_and_transfer",
        "saved_generator_sha256": _sha(parent.CANONICAL_DIRECTORY / "descriptor_A.npz"),
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_descriptor_assemblies": 0,
        "memory_coefficients_fit": 0,
        "database_roundtrip_bitwise": bool(roundtrip),
        "wall_seconds": float(time.perf_counter() - began),
    })
    gates = frozen["contract"]["stage_2_spectrum_transfer"]["pass_requires"]
    metrics["passed"] = bool(
        metrics["frequency_count_including_DC"] == gates["frequency_count_including_DC"]
        and metrics["maximum_frequency_solve_relative_residual"]
        <= gates["frequency_solve_relative_residual_max"]
        and metrics["maximum_transfer_conjugate_symmetry_relative_defect"]
        <= gates["transfer_conjugate_symmetry_relative_defect_max"]
        and metrics["complex_schur_reconstruction_relative_defect"]
        <= gates["complex_schur_reconstruction_relative_defect_max"]
        and metrics["complex_schur_unitarity_defect"]
        <= gates["complex_schur_unitarity_defect_max"]
        and metrics["database_roundtrip_bitwise"]
        and metrics["new_nonlinear_roots"] == 0
        and metrics["propagated_states"] == 0
        and metrics["new_full_560_direction_descriptor_assemblies"] == 0
        and metrics["memory_coefficients_fit"] == 0
    )
    _write_json(SCRATCH_DIRECTORY / "stage_2_metrics.json", metrics)
    return metrics


def _classification(stage_1: dict, stage_2: dict) -> tuple[str, str | None]:
    if not stage_1["passed"]:
        return STAGE_1_FAIL_CLASSIFICATION, None
    if not stage_2.get("passed", False):
        return STAGE_2_FAIL_CLASSIFICATION, None
    if stage_2["unstable_unresolved_pole_count"]:
        return UNSTABLE_CLASSIFICATION, "definitions_only_resolved_mode_promotion_manifest"
    if stage_2["cycle_scale_stable_unresolved_pole_count"]:
        return CYCLE_SCALE_CLASSIFICATION, "definitions_only_resolved_mode_promotion_manifest"
    return PASS_CLASSIFICATION, "definitions_only_mode_selection_and_finite_memory_manifest"


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
        raise RuntimeError("projection/spectrum audit is already canonicalized")
    stage_1 = _read(SCRATCH_DIRECTORY / "stage_1_metrics.json")
    stage_2_path = SCRATCH_DIRECTORY / "stage_2_metrics.json"
    stage_2 = _read(stage_2_path) if stage_2_path.exists() else {"passed": False}
    classification, authorized_next = _classification(stage_1, stage_2)
    passed = bool(stage_1["passed"] and stage_2.get("passed", False))
    unstable = int(stage_2.get("unstable_unresolved_pole_count", -1))
    cycle_scale = int(stage_2.get("cycle_scale_stable_unresolved_pole_count", -1))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "stage_1_projection_passed": stage_1["passed"],
        "stage_2_spectrum_transfer_passed": stage_2.get("passed", False),
        "resolved_dimension": stage_1["resolved_dimension"],
        "unresolved_dimension": stage_1["unresolved_dimension"],
        "unstable_unresolved_pole_count": unstable,
        "cycle_scale_stable_unresolved_pole_count": cycle_scale,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_descriptor_assemblies": 0,
        "memory_coefficients_fit": 0,
        "mode_selection_or_promotion_manifest_authorized": passed,
        "memory_fit_executed": False,
        "full_anchor_campaign_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    for name in (
        "stage_1_metrics.json",
        "projection.npz",
        "stage_2_metrics.json",
        "transfer_real.npz",
        "transfer_imag.npz",
        "unresolved_poles.npz",
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
        "saved_generator_sha256": _sha(parent.CANONICAL_DIRECTORY / "descriptor_A.npz"),
        "saved_descriptor_sha256": _sha(parent.CANONICAL_DIRECTORY / "descriptor_E.npz"),
        "saved_projection_sha256": _sha(parent.CANONICAL_DIRECTORY / "projection.npz"),
    })
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py",
        parent.THIS_RUNNER,
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
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in checksum_names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join((
            "# Invariant-compatible projection and spectrum audit WP10c9d6c7c3b5c4f25e",
            "",
            "## Classification",
            "",
            f"`{classification}`",
            "",
            "No nonlinear root or state propagation was performed, and the hash-locked complete 560-direction generator was reused without reassembly.",
            "",
            f"Stage 1 pass: `{stage_1['passed']}`. Resolved/unresolved dimensions: `{stage_1['resolved_dimension']}/{stage_1['unresolved_dimension']}`. Constraint-rowspace defect: `{stage_1['constraint_rowspace_relative_defect']:.6e}`. RL defect: `{stage_1['restriction_lifting_identity_defect']:.6e}`. Condition number: `{stage_1['resolved_condition_number']:.6e}`.",
            "",
            f"Stage 2 pass: `{stage_2.get('passed', False)}`. Frequency residual: `{stage_2.get('maximum_frequency_solve_relative_residual', float('nan')):.6e}`. Conjugacy defect: `{stage_2.get('maximum_transfer_conjugate_symmetry_relative_defect', float('nan')):.6e}`. Unstable/cycle-scale unresolved poles: `{unstable}/{cycle_scale}`.",
            "",
            f"Authorized next artifact: `{authorized_next}`. No memory fit, anchor campaign, online solver, predictive cycle, or reduced slow evolution is authorized by this audit.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "finalize", "all"), default="all"
    )
    arguments = parser.parse_args()
    if arguments.stage in {"stage1", "all"}:
        print(json.dumps(_plain(_stage_1()), indent=2, sort_keys=True), flush=True)
    if arguments.stage in {"stage2", "all"}:
        print(json.dumps(_plain(_stage_2()), indent=2, sort_keys=True), flush=True)
    if arguments.stage in {"finalize", "all"}:
        print(json.dumps(_plain(_finalize()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
