#!/usr/bin/env python3
"""Execute the 32-cell conservative coarse-PDE fallback audit."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_invariant_projection_spectrum_audit_wp10c9d6c7c3b5c4f25e as projection_tools  # noqa: E402
import run_causal_inner_larger_coarse_pde_manifest_wp10c9d6c7c3b5c4f25j as manifest  # noqa: E402
import run_causal_inner_resolved_mode_promotion_audit_wp10c9d6c7c3b5c4f25g as promotion_tools  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import _q3_physical_selectors  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    _descriptor_matrices,
    _node_reconstruction_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25k"
MANIFEST_COMMIT = "093ad83356023ff41b0031e30d390ece6ef5e941"
MANIFEST_PARENT = "e4aa3aa426d055f4b7affc6ae734e3a425586ba7"
MANIFEST_TREE = "3497c8d0de702674d83970a4b69ef0736cfcead3"

ARTIFACT = "causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k.py"
THIS_TEST = "tests/test_causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_LARGER_COARSE_PDE_"
    "AUDIT_WP10C9D6C7C3B5C4F25K_2026-08-18.md"
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

PASS_CLASSIFICATION = "R32_conservative_coarse_PDE_supported_cross_anchor_manifest_authorized"
DIMENSION_FAIL_CLASSIFICATION = "R32_promotion_exceeds_online_dimension_budget_stop"
CLOSURE_FAIL_CLASSIFICATION = "R32_no_memory_closure_insufficient_architecture_reassessment_required"
NUMERICAL_FAIL_CLASSIFICATION = "R32_conservative_projection_audit_failed_stop"


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
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return float(
        np.linalg.norm(left_array - right_array)
        / max(float(np.linalg.norm(left_array)), float(np.linalg.norm(right_array)), np.finfo(float).tiny)
    )


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("R32 manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("R32 manifest parent changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("R32 manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["R32_projection_promotion_and_no_memory_screen_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["execution_budget"]["allowed_new_nonlinear_roots"] != 0
        or contract["execution_budget"]["allowed_memory_coefficients_fit"] != 0
    ):
        raise RuntimeError("R32 authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    decisive_paths = {
        "saved_generator": manifest.GENERATOR_DIRECTORY / "descriptor_A.npz",
        "saved_descriptor": manifest.GENERATOR_DIRECTORY / "descriptor_E.npz",
        "saved_a2_output": manifest.GENERATOR_DIRECTORY / "projection.npz",
        "R16_promotion": manifest.PROMOTION_DIRECTORY / "promotion.npz",
        "compact_memory_rejection": manifest.PARENT_DIRECTORY / "metrics.json",
    }
    for name, path in decisive_paths.items():
        if _sha(path) != contract["parent_decisive_hashes"][name]:
            raise RuntimeError(f"decisive saved input changed: {path}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("R32 execution requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _R32_groups(n_cells: int) -> tuple[tuple[int, int], ...]:
    parent_groups = projection_tools.parent._coarse_groups(n_cells)
    groups = []
    for start, stop in parent_groups:
        middle = start + (stop - start) // 2
        if middle <= start or middle >= stop:
            raise RuntimeError("R16 group cannot be split into two nonempty R32 groups")
        groups.extend(((start, middle), (middle, stop)))
    result = tuple(groups)
    if len(result) != manifest.COARSE_CELLS:
        raise RuntimeError("R32 group count changed")
    if result[0][0] != 0 or result[-1][1] != n_cells:
        raise RuntimeError("R32 groups do not cover the truth grid")
    if any(left[1] != right[0] for left, right in zip(result[:-1], result[1:])):
        raise RuntimeError("R32 groups are not contiguous")
    if any(result[2 * index][0] != group[0] or result[2 * index + 1][1] != group[1] for index, group in enumerate(parent_groups)):
        raise RuntimeError("R16 boundaries are not nested in R32")
    return result


def _R32_storage_restriction(
    mapped: np.ndarray,
    height: np.ndarray,
    conservation_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapped = np.asarray(mapped, dtype=float)
    height = np.asarray(height, dtype=float)
    rows = np.asarray(conservation_rows, dtype=float)
    complete = mapped + height
    physical_mapped = C * rows.ravel()[:, None] * mapped
    physical_complete = C * rows.ravel()[:, None] * complete
    unscaled = np.zeros((manifest.STORAGE_DIMENSION, mapped.shape[1]))
    for coarse_cell, (start, stop) in enumerate(_R32_groups(rows.shape[0])):
        for field in range(manifest.FIELDS_PER_CELL):
            target = manifest.FIELDS_PER_CELL * coarse_cell + field
            source_rows = manifest.FIELDS_PER_CELL * np.arange(start, stop) + field
            source = physical_mapped if field in CONSERVATIVE_FIELDS else physical_complete
            unscaled[target] = np.sum(source[source_rows], axis=0)
    scales = np.linalg.norm(unscaled, axis=1)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise RuntimeError("R32 storage restriction lost a row")
    return unscaled / scales[:, None], unscaled, scales


def _no_memory_errors(
    transfer: np.ndarray,
    direct: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
) -> tuple[np.ndarray, dict]:
    input_scales = np.sqrt(np.sum(forcing * forcing, axis=0) + np.sum(direct * direct, axis=0))
    output_scales = np.sqrt(
        np.sum(observation * observation, axis=1) + np.sum(direct * direct, axis=1)
    )
    input_scales = np.where(input_scales > 0.0, input_scales, 1.0)
    output_scales = np.where(output_scales > 0.0, output_scales, 1.0)
    normalized = transfer / output_scales[None, :, None] / input_scales[None, None, :]
    normalized_direct = direct / output_scales[:, None] / input_scales[None, :]
    errors = np.asarray(
        [
            np.linalg.norm(normalized[index] - normalized_direct)
            / max(float(np.linalg.norm(normalized[index])), np.finfo(float).tiny)
            for index in range(normalized.shape[0])
        ]
    )
    metrics = {
        "maximum_normalized_total_transfer_relative_error": float(np.max(errors)),
        "RMS_normalized_total_transfer_relative_error": float(np.sqrt(np.mean(errors * errors))),
        "DC_normalized_total_transfer_relative_error": float(errors[0]),
    }
    return errors, metrics


def _execute() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if SCRATCH_DIRECTORY.exists():
        raise RuntimeError("R32 scratch output already exists")
    began = time.perf_counter()
    data = projection_tools.parent._seed_data()
    state = np.asarray(data["state"], dtype=float)
    context = data["context"]
    columns = np.asarray(data["columns"], dtype=float)
    rows = np.asarray(data["rows"], dtype=float)
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
    with np.load(manifest.GENERATOR_DIRECTORY / "descriptor_E.npz", allow_pickle=False) as source:
        saved_descriptor = np.asarray(source["descriptor"], dtype=float)
    with np.load(manifest.GENERATOR_DIRECTORY / "projection.npz", allow_pickle=False) as source:
        stable_dual = np.asarray(source["a2_dual"], dtype=float)
        output_map = np.asarray(source["output_map"], dtype=float)
        incidence = np.asarray(source["incidence_matrix"], dtype=float)
    with np.load(manifest.GENERATOR_DIRECTORY / "descriptor_A.npz", allow_pickle=False) as source:
        generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    storage, physical_storage, storage_scales = _R32_storage_restriction(mapped, height, rows)
    restriction, lifting, complement, projection_metrics = projection_tools._complete_qr_projection(
        storage, stable_dual
    )
    q_selectors = _q3_physical_selectors(state.shape[0], 72, rows)
    q_physical = np.asarray(q_selectors @ mapped, dtype=float)
    q_scaled = q_physical / np.linalg.norm(q_physical, axis=1)[:, None]
    constraint_defect = _relative(q_scaled @ lifting @ restriction, q_scaled)
    physical_mapped = C * rows.ravel()[:, None] * mapped
    truth_totals = []
    coarse_totals = []
    for field in CONSERVATIVE_FIELDS:
        truth_totals.append(np.sum(physical_mapped[field::manifest.FIELDS_PER_CELL], axis=0))
        coarse_totals.append(np.sum(physical_storage[field::manifest.FIELDS_PER_CELL], axis=0))
    telescope_defect = _relative(np.asarray(coarse_totals), np.asarray(truth_totals))
    descriptor_parity = _relative(mapped + height, saved_descriptor)
    margin = frozen["contract"]["ordered_schur_promotion"]["stability_margin_per_second"]
    promoted, promotion_metrics = promotion_tools._ordered_real_schur_promotion(
        generator, restriction, lifting, complement, stability_margin=margin
    )
    frequencies = None
    with np.load(manifest.PROMOTION_DIRECTORY / "transfer_real.npz", allow_pickle=False) as source:
        frequencies = np.asarray(source["angular_frequencies_per_second"], dtype=float)
    transfer, transfer_arrays, transfer_metrics = promotion_tools._stable_transfer(
        generator,
        promoted["augmented_resolved_lifting"],
        promoted["remaining_stable_truth_basis"],
        output_map,
        frequencies,
    )
    errors, closure_metrics = _no_memory_errors(
        transfer,
        transfer_arrays["augmented_direct"],
        promoted["remaining_stable_forcing"],
        transfer_arrays["stable_observation"],
    )
    SCRATCH_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(
        SCRATCH_DIRECTORY / "R32_projection_promotion.npz",
        normalized_storage_restriction=storage,
        physical_storage_restriction=physical_storage,
        storage_restriction_row_scales=storage_scales,
        resolved_restriction=restriction,
        resolved_lifting=lifting,
        augmented_resolved_restriction=promoted["augmented_resolved_restriction"],
        augmented_resolved_lifting=promoted["augmented_resolved_lifting"],
        promoted_truth_basis=promoted["promoted_truth_basis"],
        remaining_stable_truth_basis=promoted["remaining_stable_truth_basis"],
        remaining_stable_operator=promoted["remaining_stable_operator"],
        remaining_stable_forcing=promoted["remaining_stable_forcing"],
        stable_observation=transfer_arrays["stable_observation"],
        augmented_direct=transfer_arrays["augmented_direct"],
        R32_boundaries=np.asarray((0, *[stop for _, stop in _R32_groups(state.shape[0])]), dtype=int),
    )
    _write_npz(
        SCRATCH_DIRECTORY / "R32_transfer.npz",
        angular_frequencies_per_second=frequencies,
        transfer_real=np.real(transfer),
        transfer_imag=np.imag(transfer),
        no_memory_total_relative_errors=errors,
    )
    roundtrip = True
    with np.load(SCRATCH_DIRECTORY / "R32_projection_promotion.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(
            source["augmented_resolved_lifting"], promoted["augmented_resolved_lifting"]
        )
        roundtrip &= np.array_equal(
            source["remaining_stable_operator"], promoted["remaining_stable_operator"]
        )
    with np.load(SCRATCH_DIRECTORY / "R32_transfer.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["transfer_real"], np.real(transfer))
        roundtrip &= np.array_equal(source["transfer_imag"], np.imag(transfer))
        roundtrip &= np.array_equal(source["no_memory_total_relative_errors"], errors)
    metrics = {
        "stage": "R32_conservative_projection_promotion_and_no_memory_closure",
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 0,
        "seed_local_mapped_height_descriptor_reconstructions": 1,
        "memory_coefficients_fit": 0,
        "coarse_cells": manifest.COARSE_CELLS,
        "truth_dimension": int(generator.shape[0]),
        "constraint_rowspace_relative_defect": constraint_defect,
        "saved_complete_descriptor_relative_parity_defect": descriptor_parity,
        "M_J_E_telescope_relative_defect": telescope_defect,
        "node_reconstruction_relative_defect": reconstruction_defect,
        "node_partition_of_unity_defect": partition_defect,
        "database_roundtrip_bitwise": bool(roundtrip),
        "wall_seconds": float(time.perf_counter() - began),
        **projection_metrics,
        **promotion_metrics,
        **transfer_metrics,
        **closure_metrics,
    }
    algebra = frozen["contract"]["algebra_pass_requires"]
    metrics["projection_algebra_passed"] = bool(
        metrics["resolved_rank"] == algebra["resolved_rank"]
        and metrics["resolved_dimension"] == manifest.BASE_RESOLVED_DIMENSION
        and metrics["resolved_condition_number"] <= algebra["resolved_condition_number_max"]
        and metrics["restriction_lifting_identity_defect"] <= algebra["restriction_lifting_identity_max"]
        and metrics["restriction_complement_annihilation_defect"] <= algebra["restriction_complement_annihilation_max"]
        and metrics["complement_orthogonality_defect"] <= algebra["complement_orthogonality_max"]
        and metrics["constraint_rowspace_relative_defect"] <= algebra["constraint_rowspace_relative_defect_max"]
        and metrics["saved_complete_descriptor_relative_parity_defect"] <= algebra["saved_complete_descriptor_relative_parity_max"]
        and metrics["M_J_E_telescope_relative_defect"] <= algebra["M_J_E_telescope_relative_defect_max"]
    )
    promotion_contract = frozen["contract"]["ordered_schur_promotion"]
    metrics["dimension_budget_passed"] = bool(
        metrics["promoted_dimension"] <= promotion_contract["maximum_promoted_dimension"]
        and metrics["augmented_resolved_dimension"]
        <= promotion_contract["maximum_online_continuous_dimension"]
    )
    metrics["remaining_unresolved_strictly_stable"] = bool(
        metrics["remaining_unresolved_spectral_abscissa_per_second"]
        <= algebra["remaining_unresolved_spectral_abscissa_per_second_max"]
    )
    closure_gates = frozen["contract"]["no_memory_closure_screen"]["pass_requires"]
    metrics["no_memory_closure_passed"] = bool(
        metrics["maximum_normalized_total_transfer_relative_error"]
        <= closure_gates["maximum_normalized_total_transfer_relative_error_max"]
        and metrics["RMS_normalized_total_transfer_relative_error"]
        <= closure_gates["RMS_normalized_total_transfer_relative_error_max"]
        and metrics["DC_normalized_total_transfer_relative_error"]
        <= closure_gates["DC_normalized_total_transfer_relative_error_max"]
        and metrics["maximum_frequency_solve_relative_residual"]
        <= closure_gates["maximum_frequency_solve_relative_residual_max"]
        and metrics["database_roundtrip_bitwise"]
    )
    metrics["passed"] = bool(
        metrics["projection_algebra_passed"]
        and metrics["dimension_budget_passed"]
        and metrics["remaining_unresolved_strictly_stable"]
        and metrics["no_memory_closure_passed"]
        and metrics["new_nonlinear_roots"] == 0
        and metrics["propagated_states"] == 0
        and metrics["new_full_560_direction_generator_assemblies"] == 0
        and metrics["memory_coefficients_fit"] == 0
    )
    _write_json(SCRATCH_DIRECTORY / "metrics.json", metrics)
    return metrics


def _classification(metrics: dict) -> tuple[str, str | None]:
    if not metrics.get("projection_algebra_passed", False) or not metrics.get(
        "remaining_unresolved_strictly_stable", False
    ):
        return NUMERICAL_FAIL_CLASSIFICATION, None
    if not metrics.get("dimension_budget_passed", False):
        return DIMENSION_FAIL_CLASSIFICATION, None
    if not metrics.get("no_memory_closure_passed", False):
        return CLOSURE_FAIL_CLASSIFICATION, "definitions_only_reduced_architecture_reassessment_manifest"
    return PASS_CLASSIFICATION, "definitions_only_cross_anchor_closure_database_manifest"


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
        raise RuntimeError("R32 audit is already canonicalized")
    metrics = _read(SCRATCH_DIRECTORY / "metrics.json")
    classification, authorized_next = _classification(metrics)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": metrics["passed"],
        "coarse_cells": metrics["coarse_cells"],
        "base_resolved_dimension": metrics["resolved_dimension"],
        "promoted_dimension": metrics["promoted_dimension"],
        "online_continuous_dimension": metrics["augmented_resolved_dimension"],
        "remaining_unresolved_dimension": metrics["remaining_unresolved_dimension"],
        "projection_algebra_passed": metrics["projection_algebra_passed"],
        "dimension_budget_passed": metrics["dimension_budget_passed"],
        "remaining_unresolved_strictly_stable": metrics["remaining_unresolved_strictly_stable"],
        "no_memory_closure_passed": metrics["no_memory_closure_passed"],
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 0,
        "memory_coefficients_fit": 0,
        "production_coefficients_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    for name in ("metrics.json", "R32_projection_promotion.npz", "R32_transfer.npz"):
        shutil.copy2(SCRATCH_DIRECTORY / name, CANONICAL_DIRECTORY / name)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_parent": MANIFEST_PARENT,
        "manifest_tree": MANIFEST_TREE,
        "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
        "saved_generator_sha256": _sha(manifest.GENERATOR_DIRECTORY / "descriptor_A.npz"),
    })
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST)
    _write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
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
    REPORT_PATH.write_text(
        "\n".join((
            "# Larger conservative coarse-PDE audit WP10c9d6c7c3b5c4f25k",
            "",
            "## Classification",
            "",
            f"`{classification}`",
            "",
            "The saved generator was reused and one local mapped/height descriptor reconstruction built the nested 32-cell conservative restriction. No root, propagation, full generator assembly, or memory fit was performed.",
            "",
            f"Projection algebra pass: `{metrics['projection_algebra_passed']}`. Base/promoted/online dimensions: `{metrics['resolved_dimension']}/{metrics['promoted_dimension']}/{metrics['augmented_resolved_dimension']}`. Remaining stable dimension and spectral abscissa: `{metrics['remaining_unresolved_dimension']}` and `{metrics['remaining_unresolved_spectral_abscissa_per_second']:.6e} s^-1`.",
            "",
            f"No-memory closure pass: `{metrics['no_memory_closure_passed']}`. Maximum/RMS/DC normalized total-transfer errors: `{metrics['maximum_normalized_total_transfer_relative_error']:.6e}/{metrics['RMS_normalized_total_transfer_relative_error']:.6e}/{metrics['DC_normalized_total_transfer_relative_error']:.6e}`.",
            "",
            f"Authorized next artifact: `{authorized_next}`. Production coefficients, the online solver, a predictive cycle, and reduced slow evolution remain blocked.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("execute", "finalize", "all"), default="all")
    arguments = parser.parse_args()
    if arguments.stage in {"execute", "all"}:
        print(json.dumps(_plain(_execute()), indent=2, sort_keys=True), flush=True)
    if arguments.stage in {"finalize", "all"}:
        print(json.dumps(_plain(_finalize()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
