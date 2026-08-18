#!/usr/bin/env python3
"""Execute the ordered-real-Schur resolved-mode promotion audit."""

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
from scipy.linalg import schur, solve_triangular, subspace_angles


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_resolved_mode_promotion_manifest_wp10c9d6c7c3b5c4f25f as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25g"
MANIFEST_COMMIT = "caa76c5f3c5eb8385a857e0bc8cea4d2e7e30ade"
MANIFEST_PARENT = "ac265f71c9ed6ce5235e225b2ebf99677fe254ea"
MANIFEST_TREE = "dc2edbe736742a8514697374b4f682d18ea06a3e"

ARTIFACT = "causal_inner_resolved_mode_promotion_audit_wp10c9d6c7c3b5c4f25g"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_resolved_mode_promotion_audit_wp10c9d6c7c3b5c4f25g.py"
THIS_TEST = "tests/test_causal_inner_resolved_mode_promotion_audit_wp10c9d6c7c3b5c4f25g.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_RESOLVED_MODE_PROMOTION_"
    "AUDIT_WP10C9D6C7C3B5C4F25G_2026-08-18.md"
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

PASS_CLASSIFICATION = "resolved_mode_promotion_passed_stable_memory_manifest_authorized"
BUDGET_FAIL_CLASSIFICATION = "reduced_architecture_dimension_budget_failed_stop"
STABILITY_FAIL_CLASSIFICATION = "resolved_mode_promotion_failed_remaining_memory_not_stable"
NUMERICAL_FAIL_CLASSIFICATION = "resolved_mode_promotion_audit_failed_stop"


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
    scale = max(
        float(np.linalg.norm(left_array)),
        float(np.linalg.norm(right_array)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left_array - right_array) / scale)


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("resolved-mode promotion manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("resolved-mode promotion manifest parent changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("resolved-mode promotion manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["ordered_real_schur_attribution_authorized"]
        or not summary["algebraic_promotion_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["execution_budget"]["allowed_new_nonlinear_roots"] != 0
        or contract["execution_budget"]["allowed_memory_coefficients_fit"] != 0
    ):
        raise RuntimeError("resolved-mode promotion authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    decisive_paths = {
        "parent_projection": manifest.PARENT_DIRECTORY / "projection.npz",
        "parent_poles": manifest.PARENT_DIRECTORY / "unresolved_poles.npz",
        "saved_generator": manifest.GENERATOR_DIRECTORY / "descriptor_A.npz",
    }
    for name, path in decisive_paths.items():
        if _sha(path) != contract["parent_decisive_hashes"][name]:
            raise RuntimeError(f"decisive saved array changed: {path}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("resolved-mode promotion execution requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _ordered_real_schur_promotion(
    generator: np.ndarray,
    restriction: np.ndarray,
    lifting: np.ndarray,
    complement: np.ndarray,
    *,
    stability_margin: float,
) -> tuple[dict[str, np.ndarray], dict]:
    generator = np.asarray(generator, dtype=float)
    restriction = np.asarray(restriction, dtype=float)
    lifting = np.asarray(lifting, dtype=float)
    complement = np.asarray(complement, dtype=float)
    unresolved = complement.T @ generator @ complement

    def stable(real, imaginary):
        del imaginary
        return real < -stability_margin

    ordered, vectors, stable_dimension = schur(
        unresolved, output="real", sort=stable, check_finite=True
    )
    stable_vectors = vectors[:, :stable_dimension]
    promoted_vectors = vectors[:, stable_dimension:]
    stable_basis = complement @ stable_vectors
    promoted_basis = complement @ promoted_vectors
    augmented_restriction = np.vstack((restriction, promoted_basis.T))
    augmented_lifting = np.column_stack((lifting, promoted_basis))
    stable_operator = stable_basis.T @ generator @ stable_basis
    stable_forcing = stable_basis.T @ generator @ augmented_lifting
    parent_poles = np.linalg.eigvals(unresolved)
    stable_poles = np.linalg.eigvals(stable_operator)

    full_ordered, full_vectors, full_stable_dimension = schur(
        generator, output="real", sort=stable, check_finite=True
    )
    full_nonstable_basis = full_vectors[:, full_stable_dimension:]
    if promoted_basis.shape[1] and full_nonstable_basis.shape[1]:
        angles = subspace_angles(promoted_basis, full_nonstable_basis)
        capture = float(
            np.linalg.norm(full_nonstable_basis.T @ promoted_basis, ord="fro") ** 2
            / promoted_basis.shape[1]
        )
        maximum_angle = float(np.max(angles))
        minimum_angle = float(np.min(angles))
    else:
        angles = np.asarray((), dtype=float)
        capture = 0.0
        maximum_angle = math.nan
        minimum_angle = math.nan
    promoted_internal = promoted_basis.T @ generator @ promoted_basis
    promoted_invariance_residual = generator @ promoted_basis - promoted_basis @ promoted_internal
    promoted_invariance_scale = max(
        float(np.linalg.norm(generator @ promoted_basis)), np.finfo(float).tiny
    )
    metrics = {
        "truth_dimension": int(generator.shape[0]),
        "parent_resolved_dimension": int(restriction.shape[0]),
        "parent_unresolved_dimension": int(complement.shape[1]),
        "parent_stable_dimension": int(stable_dimension),
        "parent_nonstable_dimension": int(promoted_vectors.shape[1]),
        "promoted_dimension": int(promoted_basis.shape[1]),
        "augmented_resolved_dimension": int(augmented_restriction.shape[0]),
        "remaining_unresolved_dimension": int(stable_basis.shape[1]),
        "parent_spectral_abscissa_per_second": float(np.max(np.real(parent_poles))),
        "remaining_unresolved_spectral_abscissa_per_second": float(
            np.max(np.real(stable_poles))
        ),
        "ordered_schur_reconstruction_relative_defect": _relative(
            vectors @ ordered @ vectors.T, unresolved
        ),
        "ordered_schur_orthogonality_defect": float(
            np.max(np.abs(vectors.T @ vectors - np.eye(vectors.shape[1])))
        ),
        "augmented_restriction_lifting_identity_defect": float(
            np.max(
                np.abs(
                    augmented_restriction @ augmented_lifting
                    - np.eye(augmented_restriction.shape[0])
                )
            )
        ),
        "augmented_restriction_complement_annihilation_defect": float(
            np.max(np.abs(augmented_restriction @ stable_basis))
        ),
        "stable_complement_orthogonality_defect": float(
            np.max(np.abs(stable_basis.T @ stable_basis - np.eye(stable_basis.shape[1])))
        ),
        "stable_complement_lifting_annihilation_defect": float(
            np.max(np.abs(stable_basis.T @ augmented_lifting))
        ),
        "full_generator_nonstable_dimension": int(
            generator.shape[0] - full_stable_dimension
        ),
        "promoted_subspace_capture_by_full_nonstable_subspace": capture,
        "promoted_to_full_nonstable_maximum_principal_angle_radians": maximum_angle,
        "promoted_to_full_nonstable_minimum_principal_angle_radians": minimum_angle,
        "promoted_full_generator_invariance_relative_residual": float(
            np.linalg.norm(promoted_invariance_residual) / promoted_invariance_scale
        ),
        "resolved_leakage_from_promoted_basis_relative": float(
            np.linalg.norm(restriction @ generator @ promoted_basis)
            / max(float(np.linalg.norm(generator @ promoted_basis)), np.finfo(float).tiny)
        ),
    }
    arrays = {
        "augmented_resolved_restriction": augmented_restriction,
        "augmented_resolved_lifting": augmented_lifting,
        "promoted_truth_basis": promoted_basis,
        "remaining_stable_truth_basis": stable_basis,
        "remaining_stable_operator": stable_operator,
        "remaining_stable_forcing": stable_forcing,
        "ordered_parent_unresolved_schur": ordered,
        "ordered_parent_unresolved_vectors": vectors,
        "full_nonstable_truth_basis": full_nonstable_basis,
        "principal_angles_radians": angles,
        "parent_poles": parent_poles,
        "remaining_stable_poles": stable_poles,
        "full_ordered_schur": full_ordered,
    }
    return arrays, metrics


def _stable_transfer(
    generator: np.ndarray,
    augmented_lifting: np.ndarray,
    stable_basis: np.ndarray,
    output_map: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    stable_operator = stable_basis.T @ generator @ stable_basis
    forcing = stable_basis.T @ generator @ augmented_lifting
    observation = output_map @ stable_basis
    direct = output_map @ augmented_lifting
    triangular, vectors = schur(stable_operator, output="complex", check_finite=True)
    transformed_forcing = vectors.conj().T @ forcing
    identity = np.eye(stable_operator.shape[0], dtype=complex)
    values = []
    maximum_residual = 0.0
    conjugacy = []
    conjugacy_indices = {0, 1, len(frequencies) // 2, len(frequencies) - 1}
    for slot, omega in enumerate(frequencies):
        system = 1j * omega * identity - triangular
        transformed = solve_triangular(system, transformed_forcing, lower=False)
        solved = vectors @ transformed
        transfer = direct + observation @ solved
        values.append(transfer)
        original = 1j * omega * np.eye(stable_operator.shape[0]) - stable_operator
        residual = original @ solved - forcing
        scale = max(
            float(np.linalg.norm(original @ solved)),
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
            conjugacy.append(_relative(direct + observation @ negative, np.conjugate(transfer)))
    transfers = np.asarray(values)
    arrays = {
        "stable_observation": observation,
        "augmented_direct": direct,
        "stable_schur": triangular,
        "stable_schur_vectors": vectors,
    }
    metrics = {
        "frequency_count_including_DC": int(frequencies.size),
        "maximum_frequency_solve_relative_residual": maximum_residual,
        "maximum_transfer_conjugate_symmetry_relative_defect": max(conjugacy, default=0.0),
        "maximum_transfer_absolute_value": float(np.max(np.abs(transfers))),
    }
    return transfers, arrays, metrics


def _execute() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if SCRATCH_DIRECTORY.exists():
        raise RuntimeError("resolved-mode promotion scratch output already exists")
    began = time.perf_counter()
    with np.load(manifest.GENERATOR_DIRECTORY / "descriptor_A.npz", allow_pickle=False) as source:
        generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    with np.load(manifest.PARENT_DIRECTORY / "projection.npz", allow_pickle=False) as source:
        restriction = np.asarray(source["resolved_restriction"], dtype=float)
        lifting = np.asarray(source["resolved_lifting"], dtype=float)
        complement = np.asarray(source["unresolved_orthonormal_basis"], dtype=float)
        output_map = np.asarray(source["output_map"], dtype=float)
    with np.load(manifest.PARENT_DIRECTORY / "transfer_real.npz", allow_pickle=False) as source:
        frequencies = np.asarray(source["angular_frequencies_per_second"], dtype=float)
    margin = frozen["contract"]["ordered_real_schur_attribution"]["stability_margin_per_second"]
    arrays, metrics = _ordered_real_schur_promotion(
        generator, restriction, lifting, complement, stability_margin=margin
    )
    transfer, transfer_arrays, transfer_metrics = _stable_transfer(
        generator,
        arrays["augmented_resolved_lifting"],
        arrays["remaining_stable_truth_basis"],
        output_map,
        frequencies,
    )
    SCRATCH_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(
        SCRATCH_DIRECTORY / "promotion.npz",
        augmented_resolved_restriction=arrays["augmented_resolved_restriction"],
        augmented_resolved_lifting=arrays["augmented_resolved_lifting"],
        promoted_truth_basis=arrays["promoted_truth_basis"],
        remaining_stable_truth_basis=arrays["remaining_stable_truth_basis"],
        remaining_stable_operator=arrays["remaining_stable_operator"],
        remaining_stable_forcing=arrays["remaining_stable_forcing"],
        stable_observation=transfer_arrays["stable_observation"],
        augmented_direct=transfer_arrays["augmented_direct"],
    )
    _write_npz(
        SCRATCH_DIRECTORY / "spectra.npz",
        parent_pole_real_per_second=np.real(arrays["parent_poles"]),
        parent_pole_imag_per_second=np.imag(arrays["parent_poles"]),
        stable_pole_real_per_second=np.real(arrays["remaining_stable_poles"]),
        stable_pole_imag_per_second=np.imag(arrays["remaining_stable_poles"]),
        principal_angles_radians=arrays["principal_angles_radians"],
    )
    _write_npz(
        SCRATCH_DIRECTORY / "transfer_real.npz",
        angular_frequencies_per_second=frequencies,
        transfer_real=np.real(transfer),
    )
    _write_npz(
        SCRATCH_DIRECTORY / "transfer_imag.npz",
        angular_frequencies_per_second=frequencies,
        transfer_imag=np.imag(transfer),
    )
    roundtrip = True
    with np.load(SCRATCH_DIRECTORY / "promotion.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(
            source["augmented_resolved_lifting"], arrays["augmented_resolved_lifting"]
        )
        roundtrip &= np.array_equal(
            source["remaining_stable_operator"], arrays["remaining_stable_operator"]
        )
    with np.load(SCRATCH_DIRECTORY / "transfer_real.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["transfer_real"], np.real(transfer))
    with np.load(SCRATCH_DIRECTORY / "transfer_imag.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["transfer_imag"], np.imag(transfer))
    metrics.update(transfer_metrics)
    metrics.update({
        "stage": "ordered_real_schur_attribution_and_promotion",
        "saved_generator_sha256": _sha(manifest.GENERATOR_DIRECTORY / "descriptor_A.npz"),
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_descriptor_assemblies": 0,
        "memory_coefficients_fit": 0,
        "database_roundtrip_bitwise": bool(roundtrip),
        "full_generator_spectrum_diagnostic_only": True,
        "physical_instability_claim_made": False,
        "wall_seconds": float(time.perf_counter() - began),
    })
    promotion = frozen["contract"]["algebraic_promotion"]
    gates = promotion["pass_requires"]
    transfer_gates = frozen["contract"]["stable_transfer_reaudit"]["pass_requires"]
    metrics["promotion_budget_passed"] = bool(
        metrics["promoted_dimension"] <= promotion["maximum_promoted_dimension"]
        and metrics["augmented_resolved_dimension"]
        <= promotion["maximum_augmented_resolved_dimension"]
    )
    metrics["remaining_unresolved_strictly_stable"] = bool(
        metrics["remaining_unresolved_spectral_abscissa_per_second"]
        <= gates["remaining_unresolved_spectral_abscissa_per_second_max"]
    )
    metrics["passed"] = bool(
        metrics["parent_nonstable_dimension"] == gates["parent_nonstable_dimension"]
        and metrics["promoted_dimension"] == gates["promoted_dimension"]
        and metrics["promotion_budget_passed"]
        and metrics["augmented_restriction_lifting_identity_defect"]
        <= gates["augmented_restriction_lifting_identity_max"]
        and metrics["augmented_restriction_complement_annihilation_defect"]
        <= gates["augmented_restriction_complement_annihilation_max"]
        and metrics["stable_complement_orthogonality_defect"]
        <= gates["stable_complement_orthogonality_max"]
        and metrics["stable_complement_lifting_annihilation_defect"]
        <= gates["stable_complement_lifting_annihilation_max"]
        and metrics["ordered_schur_reconstruction_relative_defect"]
        <= gates["ordered_schur_reconstruction_relative_defect_max"]
        and metrics["ordered_schur_orthogonality_defect"]
        <= gates["ordered_schur_orthogonality_defect_max"]
        and metrics["remaining_unresolved_strictly_stable"]
        and metrics["frequency_count_including_DC"]
        == frozen["contract"]["stable_transfer_reaudit"]["frequency_count_including_DC"]
        and metrics["maximum_frequency_solve_relative_residual"]
        <= transfer_gates["frequency_solve_relative_residual_max"]
        and metrics["maximum_transfer_conjugate_symmetry_relative_defect"]
        <= transfer_gates["conjugate_symmetry_relative_defect_max"]
        and metrics["database_roundtrip_bitwise"]
        and metrics["new_nonlinear_roots"] == 0
        and metrics["propagated_states"] == 0
        and metrics["new_full_560_direction_descriptor_assemblies"] == 0
        and metrics["memory_coefficients_fit"] == 0
    )
    _write_json(SCRATCH_DIRECTORY / "metrics.json", metrics)
    return metrics


def _classification(metrics: dict) -> tuple[str, str | None]:
    if not metrics.get("promotion_budget_passed", False):
        return BUDGET_FAIL_CLASSIFICATION, None
    if not metrics.get("remaining_unresolved_strictly_stable", False):
        return STABILITY_FAIL_CLASSIFICATION, None
    if not metrics.get("passed", False):
        return NUMERICAL_FAIL_CLASSIFICATION, None
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
        raise RuntimeError("resolved-mode promotion audit is already canonicalized")
    metrics = _read(SCRATCH_DIRECTORY / "metrics.json")
    classification, authorized_next = _classification(metrics)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": metrics["passed"],
        "parent_resolved_dimension": metrics["parent_resolved_dimension"],
        "promoted_dimension": metrics["promoted_dimension"],
        "augmented_resolved_dimension": metrics["augmented_resolved_dimension"],
        "remaining_unresolved_dimension": metrics["remaining_unresolved_dimension"],
        "remaining_unresolved_spectral_abscissa_per_second": metrics[
            "remaining_unresolved_spectral_abscissa_per_second"
        ],
        "full_generator_nonstable_dimension_diagnostic": metrics[
            "full_generator_nonstable_dimension"
        ],
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_descriptor_assemblies": 0,
        "memory_coefficients_fit": 0,
        "physical_instability_claim_made": False,
        "memory_fit_executed": False,
        "full_anchor_campaign_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    for name in ("metrics.json", "promotion.npz", "spectra.npz", "transfer_real.npz", "transfer_imag.npz"):
        shutil.copy2(SCRATCH_DIRECTORY / name, CANONICAL_DIRECTORY / name)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_parent": MANIFEST_PARENT,
        "manifest_tree": MANIFEST_TREE,
        "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
        "parent_spectrum_package_hashes": _checksums(manifest.PARENT_DIRECTORY),
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
            "# Resolved-mode promotion audit WP10c9d6c7c3b5c4f25g",
            "",
            "## Classification",
            "",
            f"`{classification}`",
            "",
            "The hash-locked saved generator and certified R82 projection were reused. No nonlinear root, state propagation, full generator assembly, or memory fit was performed.",
            "",
            f"The ordered real Schur split found `{metrics['parent_nonstable_dimension']}` nonstable compressed coordinates. All were promoted, producing an explicit R{metrics['augmented_resolved_dimension']} state and a {metrics['remaining_unresolved_dimension']}-dimensional remaining block.",
            "",
            f"The remaining spectral abscissa is `{metrics['remaining_unresolved_spectral_abscissa_per_second']:.6e} s^-1`; strict-stability pass: `{metrics['remaining_unresolved_strictly_stable']}`. The augmented RL defect is `{metrics['augmented_restriction_lifting_identity_defect']:.6e}` and the frequency residual is `{metrics['maximum_frequency_solve_relative_residual']:.6e}`.",
            "",
            f"The full-generator nonstable dimension is recorded diagnostically as `{metrics['full_generator_nonstable_dimension']}`. Neither it nor the compressed poles are interpreted here as physical instability.",
            "",
            f"Authorized next artifact: `{authorized_next}`. Memory fitting, an anchor campaign, the online solver, a predictive cycle, and reduced slow evolution remain blocked.",
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
