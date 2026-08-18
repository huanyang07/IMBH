#!/usr/bin/env python3
"""Audit exact nonstable fibers and conservative-coordinate compatibility."""

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
import time

import numpy as np
from scipy.linalg import null_space, schur


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_unstable_exact_conservative_fiber_manifest_wp10c9d6c7c3b5c4f25t as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25u"
MANIFEST_COMMIT = "8817d178634980d8319199911f6e7973d34d6d6f"
MANIFEST_PARENT = "6119e2e1dd772c80b47bb4c1b6df2c88ec3ca78e"
MANIFEST_TREE = "bc8b193d758d44188c61645be960bbd33e621276"

PASS_CLASSIFICATION = (
    "two_anchor_unstable_exact_conservative_fiber_passed_"
    "constrained_lyapunov_stable_reduction_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "unstable_exact_conservative_fiber_failed_"
    "reduced_architecture_reassessment_required"
)
NUMERICAL_FAIL_CLASSIFICATION = (
    "unstable_exact_conservative_fiber_numerical_failure_stop"
)

ARTIFACT = (
    "causal_inner_unstable_exact_conservative_fiber_audit_"
    "wp10c9d6c7c3b5c4f25u"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_unstable_exact_conservative_fiber_audit_"
    "wp10c9d6c7c3b5c4f25u.py"
)
THIS_TEST = (
    "tests/test_causal_inner_unstable_exact_conservative_fiber_audit_"
    "wp10c9d6c7c3b5c4f25u.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_UNSTABLE_EXACT_CONSERVATIVE_"
    "FIBER_AUDIT_WP10C9D6C7C3B5C4F25U_2026-08-18.md"
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
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("unstable-exact manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("unstable-exact manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("unstable-exact manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["spectral_partition"][
            "expected_nonstable_dimension_at_each_anchor"
        ] != manifest.EXPECTED_NONSTABLE_DIMENSION
        or contract["execution_budget"][
            "allowed_new_full_560_direction_generator_assemblies"
        ] != 0
    ):
        raise RuntimeError("unstable-exact execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(manifest.PARENT_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent decisive input changed: {name}")
    saved = contract["saved_input_hashes"]
    paths = {
        "primary_generator": manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz",
        "primary_projection": manifest.PRIMARY_GENERATOR_DIRECTORY / "projection.npz",
        "heldout_generator": manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz",
        "R32_projection": manifest.R32_DIRECTORY / "R32_projection_promotion.npz",
    }
    for name, path in paths.items():
        if _sha(path) != saved[name]:
            raise RuntimeError(f"saved audit input changed: {name}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("unstable-exact audit requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(float).tiny)
    )


def _real_basis(complex_basis: np.ndarray, expected_rank: int) -> tuple[np.ndarray, dict]:
    real_span = np.hstack((np.real(complex_basis), np.imag(complex_basis)))
    left, singular, _ = np.linalg.svd(real_span, full_matrices=False)
    tolerance = max(real_span.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    basis = left[:, :expected_rank]
    defect = float(
        np.linalg.norm(complex_basis - basis @ (basis.T @ complex_basis))
        / max(float(np.linalg.norm(complex_basis)), np.finfo(float).tiny)
    )
    return basis, {
        "realification_rank": rank,
        "realification_relative_defect": defect,
        "realification_leading_singular_value": float(singular[0]),
        "realification_tail_singular_value": float(singular[expected_rank - 1]),
    }


def _spectral_fiber(
    generator: np.ndarray, threshold: float, expected_rank: int
) -> tuple[dict[str, np.ndarray], dict]:
    selector = lambda value: bool(np.real(value) >= threshold)
    right_triangular, right_schur, right_count = schur(
        generator, output="complex", sort=selector
    )
    left_triangular, left_schur, left_count = schur(
        generator.T, output="complex", sort=selector
    )
    right, right_real = _real_basis(right_schur[:, :right_count], expected_rank)
    left, left_real = _real_basis(left_schur[:, :left_count], expected_rank)
    overlap = left.T @ right
    left_dual_transpose = np.linalg.solve(overlap, left.T)
    unstable_operator = left_dual_transpose @ generator @ right
    projector = right @ left_dual_transpose
    stable_basis = null_space(left_dual_transpose)
    stable_operator = stable_basis.T @ generator @ stable_basis
    full_poles = np.linalg.eigvals(generator)
    stable_poles = np.linalg.eigvals(stable_operator)
    metrics = {
        "right_ordered_schur_count": int(right_count),
        "left_ordered_schur_count": int(left_count),
        "right_realification": right_real,
        "left_realification": left_real,
        "left_right_overlap_condition_number": float(np.linalg.cond(overlap)),
        "biorthogonality_defect": float(
            np.max(np.abs(left_dual_transpose @ right - np.eye(expected_rank)))
        ),
        "spectral_projector_idempotence_relative_defect": _relative(
            projector @ projector, projector
        ),
        "spectral_projector_commutator_relative_defect": _relative(
            generator @ projector, projector @ generator
        ),
        "right_invariance_relative_defect": _relative(
            generator @ right, right @ unstable_operator
        ),
        "left_invariance_relative_defect": _relative(
            left_dual_transpose @ generator,
            unstable_operator @ left_dual_transpose,
        ),
        "stable_complement_invariance_relative_defect": _relative(
            generator @ stable_basis, stable_basis @ stable_operator
        ),
        "stable_complement_spectral_abscissa_per_second": float(
            np.max(np.real(stable_poles))
        ),
        "full_nonstable_eigenvalue_count": int(
            np.sum(np.real(full_poles) >= threshold)
        ),
        "stable_complement_nonstable_eigenvalue_count": int(
            np.sum(np.real(stable_poles) >= threshold)
        ),
        "minimum_real_part_distance_to_partition_per_second": float(
            np.min(np.abs(np.real(full_poles) - threshold))
        ),
        "spectral_projector_operator_norm": float(np.linalg.norm(projector, 2)),
    }
    return {
        "right_basis": right,
        "left_dual_transpose": left_dual_transpose,
        "unstable_operator": unstable_operator,
        "spectral_projector": projector,
        "stable_basis": stable_basis,
        "stable_operator": stable_operator,
        "full_poles": full_poles,
        "stable_poles": stable_poles,
        "right_schur_diagonal": np.diag(right_triangular),
        "left_schur_diagonal": np.diag(left_triangular),
    }, metrics


def _matrix_rank(values: np.ndarray) -> tuple[int, np.ndarray]:
    singular = np.linalg.svd(values, compute_uv=False)
    tolerance = max(values.shape) * np.finfo(float).eps * singular[0]
    return int(np.sum(singular > tolerance)), singular


def _conservative_compatibility(
    fiber: dict[str, np.ndarray],
    restriction: np.ndarray,
    lifting: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    stable_basis = fiber["stable_basis"]
    stable_coordinate_map = restriction @ stable_basis
    stable_rank, stable_singular = _matrix_rank(stable_coordinate_map)
    stable_coordinate_lifting = np.linalg.pinv(stable_coordinate_map)
    stable_physical_lifting = stable_basis @ stable_coordinate_lifting
    unresolved_projector = np.eye(restriction.shape[1]) - lifting @ restriction
    residual = unresolved_projector @ fiber["right_basis"]
    residual_rank, residual_singular = _matrix_rank(residual)
    residual_basis = np.linalg.svd(residual, full_matrices=False)[0][:, :residual_rank]
    residual_dual = residual_basis.T @ unresolved_projector
    augmented_lifting = np.hstack((lifting, residual_basis))
    augmented_restriction = np.vstack((restriction, residual_dual))
    augmented_projector = augmented_lifting @ augmented_restriction
    metrics = {
        "R32_stable_coordinate_rank": stable_rank,
        "R32_stable_coordinate_condition_number": float(
            stable_singular[0] / stable_singular[-1]
        ),
        "stable_physical_lifting_identity_defect": float(
            np.max(
                np.abs(
                    restriction @ stable_physical_lifting
                    - np.eye(restriction.shape[0])
                )
            )
        ),
        "stable_physical_lifting_nonstable_annihilation_defect": float(
            np.max(np.abs(fiber["left_dual_transpose"] @ stable_physical_lifting))
        ),
        "nonstable_residual_rank": residual_rank,
        "nonstable_residual_condition_number": float(
            residual_singular[0] / residual_singular[residual_rank - 1]
        ),
        "augmented_coordinate_identity_defect": float(
            np.max(
                np.abs(
                    augmented_restriction @ augmented_lifting
                    - np.eye(augmented_restriction.shape[0])
                )
            )
        ),
        "augmented_nonstable_capture_relative_defect": float(
            np.linalg.norm(
                fiber["right_basis"]
                - augmented_projector @ fiber["right_basis"]
            )
            / max(float(np.linalg.norm(fiber["right_basis"])), np.finfo(float).tiny)
        ),
        "exact_unstable_augmented_dimension": int(
            restriction.shape[0] + residual_rank
        ),
        "remaining_stable_memory_budget": int(
            manifest.MAXIMUM_ONLINE_CONTINUOUS_DIMENSION
            - restriction.shape[0]
            - residual_rank
        ),
    }
    return {
        "stable_physical_lifting": stable_physical_lifting,
        "nonstable_residual_basis": residual_basis,
        "augmented_lifting": augmented_lifting,
        "augmented_restriction": augmented_restriction,
    }, metrics


def _anchor_passed(metrics: dict, gates: dict) -> bool:
    spectral = metrics["spectral"]
    physical = metrics["conservative_compatibility"]
    return bool(
        spectral["right_ordered_schur_count"]
        == gates["selected_nonstable_dimension_equal"]
        and spectral["left_ordered_schur_count"]
        == gates["selected_nonstable_dimension_equal"]
        and spectral["full_nonstable_eigenvalue_count"]
        == gates["selected_nonstable_dimension_equal"]
        and spectral["stable_complement_nonstable_eigenvalue_count"] == 0
        and spectral["right_realification"]["realification_rank"]
        == gates["selected_nonstable_dimension_equal"]
        and spectral["left_realification"]["realification_rank"]
        == gates["selected_nonstable_dimension_equal"]
        and max(
            spectral["right_realification"]["realification_relative_defect"],
            spectral["left_realification"]["realification_relative_defect"],
        ) <= gates["realification_relative_defect_max"]
        and spectral["left_right_overlap_condition_number"]
        <= gates["left_right_overlap_condition_number_max"]
        and spectral["biorthogonality_defect"]
        <= gates["biorthogonality_defect_max"]
        and spectral["spectral_projector_idempotence_relative_defect"]
        <= gates["spectral_projector_idempotence_relative_defect_max"]
        and spectral["spectral_projector_commutator_relative_defect"]
        <= gates["spectral_projector_commutator_relative_defect_max"]
        and spectral["right_invariance_relative_defect"]
        <= gates["right_invariance_relative_defect_max"]
        and spectral["left_invariance_relative_defect"]
        <= gates["left_invariance_relative_defect_max"]
        and spectral["stable_complement_invariance_relative_defect"]
        <= gates["stable_complement_invariance_relative_defect_max"]
        and spectral["stable_complement_spectral_abscissa_per_second"]
        <= gates["stable_complement_spectral_abscissa_per_second_max"]
        and physical["R32_stable_coordinate_rank"]
        == gates["R32_stable_coordinate_rank_equal"]
        and physical["R32_stable_coordinate_condition_number"]
        <= gates["R32_stable_coordinate_condition_number_max"]
        and physical["stable_physical_lifting_identity_defect"]
        <= gates["stable_physical_lifting_identity_defect_max"]
        and physical["stable_physical_lifting_nonstable_annihilation_defect"]
        <= gates["stable_physical_lifting_nonstable_annihilation_defect_max"]
        and physical["nonstable_residual_rank"]
        <= gates["nonstable_residual_rank_max"]
        and physical["augmented_nonstable_capture_relative_defect"]
        <= gates["augmented_nonstable_capture_relative_defect_max"]
        and physical["remaining_stable_memory_budget"]
        >= gates["remaining_stable_memory_budget_min"]
    )


def _cross_anchor_metrics(fibers: dict[str, dict[str, np.ndarray]]) -> tuple[dict, dict]:
    primary_right = fibers["primary"]["right_basis"]
    heldout_right = fibers["heldout"]["right_basis"]
    primary_left = fibers["primary"]["left_dual_transpose"].T
    heldout_left = fibers["heldout"]["left_dual_transpose"].T
    right_cosines = np.linalg.svd(primary_right.T @ heldout_right, compute_uv=False)
    left_cosines = np.linalg.svd(
        np.linalg.qr(primary_left)[0].T @ np.linalg.qr(heldout_left)[0],
        compute_uv=False,
    )
    align_left, _, align_right = np.linalg.svd(heldout_right.T @ primary_right)
    rotation = align_left @ align_right
    aligned_right = heldout_right @ rotation
    aligned_left_dual_transpose = rotation.T @ fibers["heldout"]["left_dual_transpose"]
    return {
        "right_principal_cosine_min": float(np.min(right_cosines)),
        "left_principal_cosine_min": float(np.min(left_cosines)),
        "right_largest_principal_angle_degrees": float(
            np.degrees(np.arccos(np.clip(np.min(right_cosines), -1.0, 1.0)))
        ),
        "left_largest_principal_angle_degrees": float(
            np.degrees(np.arccos(np.clip(np.min(left_cosines), -1.0, 1.0)))
        ),
        "aligned_heldout_biorthogonality_defect": float(
            np.max(
                np.abs(
                    aligned_left_dual_transpose @ aligned_right
                    - np.eye(primary_right.shape[1])
                )
            )
        ),
    }, {
        "right_principal_cosines": right_cosines,
        "left_principal_cosines": left_cosines,
        "heldout_alignment_rotation": rotation,
        "heldout_aligned_right_basis": aligned_right,
        "heldout_aligned_left_dual_transpose": aligned_left_dual_transpose,
    }


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


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("unstable-exact audit is already canonicalized")
    began = time.perf_counter()
    with np.load(
        manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz", allow_pickle=False
    ) as source:
        primary_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    with np.load(
        manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz", allow_pickle=False
    ) as source:
        heldout_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    with np.load(
        manifest.R32_DIRECTORY / "R32_projection_promotion.npz", allow_pickle=False
    ) as source:
        restriction = np.asarray(source["resolved_restriction"], dtype=float)
        lifting = np.asarray(source["resolved_lifting"], dtype=float)
    generators = {"primary": primary_generator, "heldout": heldout_generator}
    threshold = frozen["contract"]["spectral_partition"][
        "nonstable_threshold_per_second"
    ]
    expected = frozen["contract"]["spectral_partition"][
        "expected_nonstable_dimension_at_each_anchor"
    ]
    gates = frozen["contract"]["binding_gates"]
    fibers = {}
    compatibility = {}
    anchor_metrics = {}
    arrays = {}
    finite = True
    for anchor, generator in generators.items():
        fibers[anchor], spectral_metrics = _spectral_fiber(
            generator, threshold, expected
        )
        compatibility[anchor], physical_metrics = _conservative_compatibility(
            fibers[anchor], restriction, lifting
        )
        anchor_metrics[anchor] = {
            "spectral": spectral_metrics,
            "conservative_compatibility": physical_metrics,
        }
        anchor_metrics[anchor]["passed"] = _anchor_passed(
            anchor_metrics[anchor], gates
        )
        for group in (fibers[anchor], compatibility[anchor]):
            for name, value in group.items():
                if name in ("stable_basis", "stable_operator"):
                    continue
                arrays[f"{anchor}_{name}"] = value
                finite &= bool(np.all(np.isfinite(value)))
    cross_metrics, cross_arrays = _cross_anchor_metrics(fibers)
    arrays.update(cross_arrays)
    cross_passed = bool(
        cross_metrics["right_principal_cosine_min"]
        >= gates["cross_anchor_right_principal_cosine_min"]
        and cross_metrics["left_principal_cosine_min"]
        >= gates["cross_anchor_left_principal_cosine_min"]
        and cross_metrics["aligned_heldout_biorthogonality_defect"]
        <= gates["biorthogonality_defect_max"]
    )
    elapsed = float(time.perf_counter() - began)
    numerical_passed = bool(
        finite
        and np.isfinite(elapsed)
        and elapsed
        <= 60.0 * frozen["contract"]["execution_budget"]["maximum_wall_minutes"]
    )
    passed = bool(
        numerical_passed
        and cross_passed
        and all(metrics["passed"] for metrics in anchor_metrics.values())
    )
    if not numerical_passed:
        classification = NUMERICAL_FAIL_CLASSIFICATION
        authorized_next = None
    elif passed:
        classification = PASS_CLASSIFICATION
        authorized_next = (
            "definitions_only_constrained_lyapunov_stable_reduction_manifest"
        )
    else:
        classification = FAIL_CLASSIFICATION
        authorized_next = (
            "definitions_only_unstable_exact_architecture_reassessment_manifest"
        )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {
        "anchors": anchor_metrics,
        "cross_anchor": cross_metrics,
        "cross_anchor_passed": cross_passed,
        "numerical_passed": numerical_passed,
        "wall_seconds": elapsed,
    })
    _write_npz(CANONICAL_DIRECTORY / "decisive_fibers.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "numerical_passed": numerical_passed,
        "cross_anchor_passed": cross_passed,
        "primary_passed": anchor_metrics["primary"]["passed"],
        "heldout_passed": anchor_metrics["heldout"]["passed"],
        "primary_nonstable_dimension": anchor_metrics["primary"]["spectral"]["full_nonstable_eigenvalue_count"],
        "heldout_nonstable_dimension": anchor_metrics["heldout"]["spectral"]["full_nonstable_eigenvalue_count"],
        "maximum_exact_unstable_augmented_dimension": max(
            metrics["conservative_compatibility"]["exact_unstable_augmented_dimension"]
            for metrics in anchor_metrics.values()
        ),
        "minimum_remaining_stable_memory_budget": min(
            metrics["conservative_compatibility"]["remaining_stable_memory_budget"]
            for metrics in anchor_metrics.values()
        ),
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 0,
        "new_truth_anchors": 0,
        "physical_failure_detected": False,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_parent": MANIFEST_PARENT,
        "manifest_tree": MANIFEST_TREE,
        "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
        "parent_package_hashes": _checksums(manifest.PARENT_DIRECTORY),
        "primary_generator_package_hashes": _checksums(manifest.PRIMARY_GENERATOR_DIRECTORY),
        "cross_anchor_package_hashes": _checksums(manifest.CROSS_ANCHOR_DIRECTORY),
        "R32_package_hashes": _checksums(manifest.R32_DIRECTORY),
    })
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST)
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
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text("\n".join((
        "# Unstable-exact conservative-fiber audit WP10c9d6c7c3b5c4f25u",
        "",
        "## Classification",
        "",
        f"`{classification}`",
        "",
        "This saved-generator audit separated the complete nonstable spectral fiber exactly before any stable reduction. It executed no truth assembly, nonlinear root, or propagation.",
        "",
        f"The primary and held-out nonstable dimensions are `{summary['primary_nonstable_dimension']}` and `{summary['heldout_nonstable_dimension']}`. The maximum conservative-plus-exact-fiber dimension is `{summary['maximum_exact_unstable_augmented_dimension']}`, leaving at least `{summary['minimum_remaining_stable_memory_budget']}` states under the R320 cap.",
        "",
        f"The largest right/left cross-anchor principal angles are `{cross_metrics['right_largest_principal_angle_degrees']:.6f}` and `{cross_metrics['left_largest_principal_angle_degrees']:.6f}` degrees.",
        "",
        f"Authorized next artifact: `{authorized_next}`. An online integrator, predictive cycle, and reduced slow evolution remain blocked.",
        "",
    )), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
