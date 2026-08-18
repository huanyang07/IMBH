#!/usr/bin/env python3
"""Execute the square-root conservative transfer-seeded saved-generator audit."""

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
from scipy.linalg import (
    cholesky,
    null_space,
    solve_continuous_lyapunov,
    solve_triangular,
)


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_complete_resolved_closure_audit_wp10c9d6c7c3b5c4f25s as complete_tools  # noqa: E402
import run_causal_inner_constrained_lyapunov_reduction_audit_wp10c9d6c7c3b5c4f25y as prior_tools  # noqa: E402
import run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m as memory_tools  # noqa: E402
import run_causal_inner_square_root_transfer_seeded_manifest_wp10c9d6c7c3b5c4f25z as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25aa"
MANIFEST_COMMIT = "90da0bc0915e9846329519e1fdd62b8a2ce81b9a"
MANIFEST_PARENT = "b4944b7e1ae0d95cd778aa883064d4c9a119819f"
MANIFEST_TREE = "7a8b917f5406ea4194985d6c3ebbd78430bcefb6"

PASS_CLASSIFICATION = (
    "two_anchor_square_root_transfer_seeded_reduction_passed_"
    "parametric_alignment_manifest_authorized"
)
CAP_FAIL_CLASSIFICATION = (
    "square_root_transfer_seeded_reduction_failed_within_R320_"
    "structured_basis_reassessment_required"
)
NUMERICAL_FAIL_CLASSIFICATION = (
    "square_root_transfer_seeded_reduction_numerical_failure_stop"
)

ARTIFACT = (
    "causal_inner_square_root_transfer_seeded_audit_"
    "wp10c9d6c7c3b5c4f25aa"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_square_root_transfer_seeded_audit_"
    "wp10c9d6c7c3b5c4f25aa.py"
)
THIS_TEST = (
    "tests/test_causal_inner_square_root_transfer_seeded_audit_"
    "wp10c9d6c7c3b5c4f25aa.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SQUARE_ROOT_TRANSFER_SEEDED_"
    "AUDIT_WP10C9D6C7C3B5C4F25AA_2026-08-18.md"
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
        raise RuntimeError("square-root manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("square-root manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("square-root manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["execution_budget"]["candidate_hidden_orders"]
        != list(manifest.HIDDEN_ORDERS)
        or contract["execution_budget"]
        ["allowed_new_full_560_direction_generator_assemblies"]
        != 0
        or not contract["authority"]["preserve_f25y_rejection"]
    ):
        raise RuntimeError("square-root execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["prior_decisive_hashes"].items():
        if _sha(manifest.PRIOR_DIRECTORY / name) != expected:
            raise RuntimeError(f"prior input changed: {name}")
    for name, expected in contract["fiber_decisive_hashes"].items():
        if _sha(manifest.FIBER_DIRECTORY / name) != expected:
            raise RuntimeError(f"fiber input changed: {name}")
    saved_paths = {
        "primary_generator": manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz",
        "primary_output": manifest.PRIMARY_GENERATOR_DIRECTORY / "projection.npz",
        "heldout_generator_and_output": manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz",
        "R32_projection": manifest.R32_DIRECTORY / "R32_projection_promotion.npz",
        "frequency_ladder": manifest.R32_DIRECTORY / "R32_transfer.npz",
        "R196_common_basis": manifest.COMMON_BASIS_DIRECTORY / "decisive_basis.npz",
    }
    for name, path in saved_paths.items():
        if _sha(path) != contract["saved_input_hashes"][name]:
            raise RuntimeError(f"saved input changed: {name}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("square-root audit requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(
            float(np.linalg.norm(left)),
            float(np.linalg.norm(right)),
            np.finfo(float).tiny,
        )
    )


def _square_root_stable_system(
    generator: np.ndarray,
    output: np.ndarray,
    restriction: np.ndarray,
    right_unstable: np.ndarray,
    left_unstable_transpose: np.ndarray,
    unstable_operator: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    stable_basis = null_space(left_unstable_transpose)
    stable_operator = stable_basis.T @ generator @ stable_basis
    stable_output = output @ stable_basis
    dimension = stable_operator.shape[0]
    certificate = solve_continuous_lyapunov(
        stable_operator.T, -np.eye(dimension)
    )
    certificate = 0.5 * (certificate + certificate.T)
    certificate_residual = (
        stable_operator.T @ certificate
        + certificate @ stable_operator
        + np.eye(dimension)
    )
    square_root = cholesky(certificate, lower=False)
    inverse_square_root = solve_triangular(
        square_root, np.eye(dimension), lower=False
    )
    whitened_operator = square_root @ stable_operator @ inverse_square_root
    whitened_output = stable_output @ inverse_square_root
    whitened_rhs = inverse_square_root.T @ inverse_square_root
    whitened_lyapunov_residual = (
        whitened_operator.T + whitened_operator + whitened_rhs
    )
    conservative_map = restriction @ stable_basis @ inverse_square_root
    left, singular_values, right_transpose = np.linalg.svd(
        conservative_map, full_matrices=True
    )
    q = restriction.shape[0]
    conservative_lift = (
        right_transpose[:q, :].T
        * (1.0 / singular_values)[None, :]
    ) @ left.T
    hidden_basis = right_transpose[q:, :].T
    full_trial = np.hstack((conservative_lift, hidden_basis))
    full_test = np.hstack((conservative_map.T, hidden_basis))
    full_reduced = full_test.T @ whitened_operator @ full_trial
    hidden_operator = full_reduced[q:, q:]
    hidden_forcing = full_reduced[q:, :q]
    combined_observation = np.vstack(
        (full_reduced[:q, q:], whitened_output @ hidden_basis)
    )
    combined_direct = np.vstack(
        (full_reduced[:q, :q], whitened_output @ conservative_lift)
    )
    metrics = {
        "stable_Lyapunov_relative_residual": float(
            np.linalg.norm(certificate_residual) / np.sqrt(dimension)
        ),
        "stable_Lyapunov_minimum_eigenvalue": float(
            np.min(np.linalg.eigvalsh(certificate))
        ),
        "stable_Lyapunov_condition_number": float(np.linalg.cond(certificate)),
        "square_root_reconstruction_relative_defect": _relative(
            square_root.T @ square_root, certificate
        ),
        "whitened_Lyapunov_relative_defect": float(
            np.linalg.norm(whitened_lyapunov_residual)
            / max(float(np.linalg.norm(whitened_rhs)), np.finfo(float).tiny)
        ),
        "conservative_map_rank": int(np.linalg.matrix_rank(conservative_map)),
        "conservative_map_largest_singular_value": float(singular_values[0]),
        "conservative_map_smallest_singular_value": float(singular_values[-1]),
        "conservative_lift_identity_defect": float(
            np.max(np.abs(conservative_map @ conservative_lift - np.eye(q)))
        ),
        "full_conservative_test_identity_defect": float(
            np.max(np.abs(full_test[:, :q] - conservative_map.T))
        ),
        "full_trial_test_biorthogonality_defect": float(
            np.max(np.abs(full_test.T @ full_trial - np.eye(dimension)))
        ),
        "full_coordinate_reconstruction_relative_defect": _relative(
            full_trial @ full_test.T, np.eye(dimension)
        ),
        "full_stable_spectral_abscissa_per_second": float(
            np.max(np.real(np.linalg.eigvals(full_reduced)))
        ),
    }
    return {
        "stable_basis": stable_basis,
        "stable_operator": stable_operator,
        "certificate": certificate,
        "square_root": square_root,
        "inverse_square_root": inverse_square_root,
        "whitened_operator": whitened_operator,
        "whitened_output": whitened_output,
        "whitened_rhs": whitened_rhs,
        "conservative_map": conservative_map,
        "conservative_lift": conservative_lift,
        "hidden_basis": hidden_basis,
        "hidden_operator": hidden_operator,
        "hidden_forcing": hidden_forcing,
        "combined_observation": combined_observation,
        "combined_direct": combined_direct,
        "right_unstable": right_unstable,
        "left_unstable_transpose": left_unstable_transpose,
        "unstable_operator": unstable_operator,
    }, metrics


def _balanced_trial(balanced: dict[str, np.ndarray], order: int) -> np.ndarray:
    hankel = balanced["hankel_singular_values"]
    if order > hankel.size or np.any(hankel[:order] <= 0.0):
        raise RuntimeError(f"transfer seed order {order} exceeds positive balanced rank")
    right = balanced["hankel_right_vectors_transpose"].T[:, :order]
    return balanced["controllability_factor"] @ (
        right * (1.0 / np.sqrt(hankel[:order]))[None, :]
    )


def _projected_transfer_seed(
    system: dict[str, np.ndarray],
    generator: np.ndarray,
    output: np.ndarray,
    augmented_restriction: np.ndarray,
    augmented_lifting: np.ndarray,
    frequencies: np.ndarray,
    heldout_frequencies: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    blocks, coordinate_metrics = complete_tools._complete_blocks(
        generator, augmented_restriction, augmented_lifting, output
    )
    prepared = complete_tools._prepare_memory(
        blocks, frequencies, heldout_frequencies
    )
    old_trial = blocks["stable_basis"] @ _balanced_trial(
        prepared["balanced"], order
    )
    exact_stable_trial = old_trial - system["right_unstable"] @ (
        system["left_unstable_transpose"] @ old_trial
    )
    whitened_trial = system["square_root"] @ (
        system["stable_basis"].T @ exact_stable_trial
    )
    hidden_coordinates = system["hidden_basis"].T @ whitened_trial
    seed_vectors, singular_values, _ = np.linalg.svd(
        hidden_coordinates, full_matrices=False
    )
    effective_rank = int(np.linalg.matrix_rank(hidden_coordinates))
    metrics = {
        "source_coordinate_reconstruction_relative_defect": coordinate_metrics[
            "coordinate_reconstruction_relative_defect"
        ],
        "source_balanced_controllability_gramian_relative_residual": prepared[
            "full_metrics"
        ]["controllability_gramian_relative_residual"],
        "source_balanced_observability_gramian_relative_residual": prepared[
            "full_metrics"
        ]["observability_gramian_relative_residual"],
        "projected_seed_effective_rank": effective_rank,
        "projected_seed_largest_singular_value": float(singular_values[0]),
        "projected_seed_smallest_singular_value": float(singular_values[-1]),
        "projected_seed_relative_singular_value_at_130": float(
            singular_values[order - 1] / singular_values[0]
        ),
    }
    return seed_vectors, singular_values, metrics


def _prepare_reference(
    system: dict[str, np.ndarray],
    frequencies: np.ndarray,
    heldout_frequencies: np.ndarray,
) -> dict:
    forcing, observation, direct, input_scales, output_scales = (
        memory_tools._normalize_system(
            system["hidden_forcing"],
            system["combined_observation"],
            system["combined_direct"],
        )
    )
    training_reference, training_residual = memory_tools._frequency_response(
        system["hidden_operator"],
        forcing,
        observation,
        direct,
        frequencies,
    )
    heldout_reference, heldout_residual = memory_tools._frequency_response(
        system["hidden_operator"],
        forcing,
        observation,
        direct,
        heldout_frequencies,
    )
    return {
        "input_scales": input_scales,
        "output_scales": output_scales,
        "direct": direct,
        "training_reference": training_reference,
        "heldout_reference": heldout_reference,
        "reference_frequency_residual": max(training_residual, heldout_residual),
    }


def _block_metrics(
    approximation: np.ndarray,
    reference: np.ndarray,
    direct: np.ndarray,
    row_slice: slice,
) -> tuple[dict, dict[str, np.ndarray]]:
    dynamic, total, metrics = memory_tools._error_metrics(
        approximation[:, row_slice], reference[:, row_slice], direct[row_slice]
    )
    return metrics, {"dynamic_errors": dynamic, "total_errors": total}


def _block_pass(metrics: dict, gates: dict) -> bool:
    return bool(
        all(
            metrics[f"{prefix}_{name.removesuffix('_max')}"] <= maximum
            for prefix in ("training", "heldout")
            for name, maximum in gates.items()
        )
    )


def _candidate(
    system: dict[str, np.ndarray],
    prepared: dict,
    seed_vectors: np.ndarray,
    order: int,
    frequencies: np.ndarray,
    heldout_frequencies: np.ndarray,
    gates: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    q = manifest.PHYSICAL_DIMENSION
    hidden_trial = system["hidden_basis"] @ seed_vectors[:, :order]
    trial = np.hstack((system["conservative_lift"], hidden_trial))
    test = np.hstack((system["conservative_map"].T, hidden_trial))
    gram = trial.T @ trial
    reduced = test.T @ system["whitened_operator"] @ trial
    hidden_operator = reduced[q:, q:]
    hidden_forcing = (
        reduced[q:, :q] / prepared["input_scales"][None, :]
    )
    observation_physical = np.vstack(
        (reduced[:q, q:], system["whitened_output"] @ hidden_trial)
    )
    direct_physical = np.vstack(
        (
            reduced[:q, :q],
            system["whitened_output"] @ system["conservative_lift"],
        )
    )
    observation = observation_physical / prepared["output_scales"][:, None]
    direct = (
        direct_physical
        / prepared["output_scales"][:, None]
        / prepared["input_scales"][None, :]
    )
    training, training_residual = memory_tools._frequency_response(
        hidden_operator, hidden_forcing, observation, direct, frequencies
    )
    heldout_response, heldout_residual = memory_tools._frequency_response(
        hidden_operator,
        hidden_forcing,
        observation,
        direct,
        heldout_frequencies,
    )
    blocks = {}
    errors = {}
    for label, row_slice in (
        ("resolved_self_energy", slice(0, q)),
        ("conservative_face_flux", slice(q, None)),
    ):
        training_metrics, training_errors = _block_metrics(
            training,
            prepared["training_reference"],
            prepared["direct"],
            row_slice,
        )
        heldout_metrics, heldout_errors = _block_metrics(
            heldout_response,
            prepared["heldout_reference"],
            prepared["direct"],
            row_slice,
        )
        blocks[label] = {
            **{f"training_{key}": value for key, value in training_metrics.items()},
            **{f"heldout_{key}": value for key, value in heldout_metrics.items()},
        }
        for prefix, values in (
            ("training", training_errors),
            ("heldout", heldout_errors),
        ):
            for name, value in values.items():
                errors[f"{label}_{prefix}_{name}"] = value
    lyapunov_target = trial.T @ (
        system["whitened_operator"] + system["whitened_operator"].T
    ) @ trial
    lyapunov_residual = (
        gram @ reduced + reduced.T @ gram - lyapunov_target
    )
    reduced_poles = np.linalg.eigvals(reduced)
    unstable_poles = np.linalg.eigvals(system["unstable_operator"])
    complete_poles = np.concatenate((unstable_poles, reduced_poles))
    threshold = manifest.NONSTABLE_THRESHOLD_PER_SECOND
    complete_nonstable = complete_poles[np.real(complete_poles) >= threshold]
    metrics = {
        "hidden_order": order,
        "conservative_test_identity_defect": float(
            np.max(np.abs(test[:, :q] - system["conservative_map"].T))
        ),
        "trial_test_biorthogonality_defect": float(
            np.max(np.abs(test.T @ trial - np.eye(q + order)))
        ),
        "hidden_conservative_annihilation_defect": float(
            np.max(np.abs(system["conservative_map"] @ hidden_trial))
        ),
        "reduced_Lyapunov_identity_relative_defect": float(
            np.linalg.norm(lyapunov_residual)
            / max(float(np.linalg.norm(lyapunov_target)), np.finfo(float).tiny)
        ),
        "reduced_stable_spectral_abscissa_per_second": float(
            np.max(np.real(reduced_poles))
        ),
        "complete_nonstable_eigenvalue_count": int(complete_nonstable.size),
        "extra_nonstable_eigenvalue_count": int(
            np.sum(np.real(reduced_poles) >= threshold)
        ),
        "exact_nonstable_pole_relative_defect": prior_tools._pole_defect(
            unstable_poles, complete_nonstable
        ),
        "maximum_frequency_solve_relative_residual": max(
            prepared["reference_frequency_residual"],
            training_residual,
            heldout_residual,
        ),
        "blocks": blocks,
    }
    metrics["numerical_passed"] = bool(
        metrics["conservative_test_identity_defect"]
        <= gates["conservative_test_identity_defect_max"]
        and metrics["trial_test_biorthogonality_defect"]
        <= gates["trial_test_biorthogonality_defect_max"]
        and metrics["hidden_conservative_annihilation_defect"]
        <= gates["hidden_conservative_annihilation_defect_max"]
        and metrics["reduced_Lyapunov_identity_relative_defect"]
        <= gates["reduced_Lyapunov_identity_relative_defect_max"]
        and metrics["maximum_frequency_solve_relative_residual"]
        <= gates["maximum_frequency_solve_relative_residual_max"]
    )
    metrics["spectral_passed"] = bool(
        metrics["reduced_stable_spectral_abscissa_per_second"]
        <= gates["reduced_stable_spectral_abscissa_per_second_max"]
        and metrics["complete_nonstable_eigenvalue_count"]
        == gates["complete_nonstable_eigenvalue_count_equal"]
        and metrics["extra_nonstable_eigenvalue_count"]
        <= gates["extra_nonstable_eigenvalue_count_max"]
        and metrics["exact_nonstable_pole_relative_defect"]
        <= gates["exact_nonstable_pole_relative_defect_max"]
    )
    metrics["transfer_passed"] = bool(
        all(_block_pass(blocks[label], gates[label]) for label in blocks)
    )
    metrics["passed"] = bool(
        metrics["numerical_passed"]
        and metrics["spectral_passed"]
        and metrics["transfer_passed"]
    )
    return metrics, {
        "hidden_truth_trial": (
            system["stable_basis"]
            @ system["inverse_square_root"]
            @ hidden_trial
        ),
        "stable_reduced_operator": reduced,
        "whitened_trial": trial,
        "whitened_test": test,
        "metric": gram,
        "normalized_hidden_forcing": hidden_forcing,
        "normalized_combined_observation": observation,
        "normalized_combined_direct": direct,
        **errors,
    }


def _base_pass(metrics: dict, seed: dict, gates: dict) -> bool:
    return bool(
        metrics["stable_Lyapunov_relative_residual"]
        <= gates["stable_Lyapunov_relative_residual_max"]
        and metrics["stable_Lyapunov_minimum_eigenvalue"]
        >= gates["stable_Lyapunov_minimum_eigenvalue_min"]
        and metrics["stable_Lyapunov_condition_number"]
        <= gates["stable_Lyapunov_condition_number_max"]
        and metrics["square_root_reconstruction_relative_defect"]
        <= gates["square_root_reconstruction_relative_defect_max"]
        and metrics["whitened_Lyapunov_relative_defect"]
        <= gates["whitened_Lyapunov_relative_defect_max"]
        and metrics["conservative_map_rank"]
        == gates["conservative_map_rank_equal"]
        and metrics["conservative_lift_identity_defect"]
        <= gates["conservative_lift_identity_defect_max"]
        and metrics["full_conservative_test_identity_defect"]
        <= gates["conservative_test_identity_defect_max"]
        and metrics["full_trial_test_biorthogonality_defect"]
        <= gates["trial_test_biorthogonality_defect_max"]
        and metrics["full_coordinate_reconstruction_relative_defect"]
        <= gates["full_coordinate_reconstruction_relative_defect_max"]
        and metrics["full_stable_spectral_abscissa_per_second"]
        <= gates["reduced_stable_spectral_abscissa_per_second_max"]
        and seed["projected_seed_effective_rank"]
        >= gates["projected_seed_effective_rank_min"]
    )


def _candidate_score(item: dict, gates: dict) -> float:
    cosine = item["cross_anchor_hidden_principal_cosine_min"]
    ratios = [
        gates["cross_anchor_hidden_principal_cosine_min"]
        / max(cosine, np.finfo(float).tiny)
    ]
    for anchor in ("primary", "heldout"):
        metrics = item[anchor]
        ratios.extend(
            (
                metrics["trial_test_biorthogonality_defect"]
                / gates["trial_test_biorthogonality_defect_max"],
                metrics["reduced_Lyapunov_identity_relative_defect"]
                / gates["reduced_Lyapunov_identity_relative_defect_max"],
            )
        )
        for block in ("resolved_self_energy", "conservative_face_flux"):
            for prefix in ("training", "heldout"):
                for name, limit in gates[block].items():
                    ratios.append(
                        metrics["blocks"][block]
                        [f"{prefix}_{name.removesuffix('_max')}"]
                        / limit
                    )
    return float(max(ratios))


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
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
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
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("square-root audit is already canonicalized")
    began = time.perf_counter()
    with np.load(
        manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz",
        allow_pickle=False,
    ) as source:
        primary_generator = np.asarray(
            source["complete_fixed_Q_generator"], dtype=float
        )
    with np.load(
        manifest.PRIMARY_GENERATOR_DIRECTORY / "projection.npz",
        allow_pickle=False,
    ) as source:
        primary_output = np.asarray(source["output_map"], dtype=float)
    with np.load(
        manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz",
        allow_pickle=False,
    ) as source:
        heldout_generator = np.asarray(
            source["complete_fixed_Q_generator"], dtype=float
        )
        heldout_output = np.asarray(source["output_map"], dtype=float)
    with np.load(
        manifest.R32_DIRECTORY / "R32_projection_promotion.npz",
        allow_pickle=False,
    ) as source:
        restriction = np.asarray(source["resolved_restriction"], dtype=float)
    with np.load(
        manifest.R32_DIRECTORY / "R32_transfer.npz", allow_pickle=False
    ) as source:
        frequencies = np.asarray(
            source["angular_frequencies_per_second"], dtype=float
        )
    heldout_frequencies = memory_tools._heldout_frequencies(frequencies)
    with np.load(
        manifest.FIBER_DIRECTORY / "decisive_fibers.npz", allow_pickle=False
    ) as source:
        fiber_arrays = {name: np.asarray(source[name]) for name in source.files}
    with np.load(
        manifest.COMMON_BASIS_DIRECTORY / "decisive_basis.npz",
        allow_pickle=False,
    ) as source:
        common_arrays = {name: np.asarray(source[name]) for name in source.files}
    generators = {
        "primary": primary_generator,
        "heldout": heldout_generator,
    }
    outputs = {"primary": primary_output, "heldout": heldout_output}
    gates = frozen["contract"]["binding_gates"]
    systems = {}
    base_metrics = {}
    seed_metrics = {}
    seeds = {}
    prepared = {}
    base_passed = True
    for anchor in ("primary", "heldout"):
        systems[anchor], base_metrics[anchor] = _square_root_stable_system(
            generators[anchor],
            outputs[anchor],
            restriction,
            fiber_arrays[f"{anchor}_right_basis"],
            fiber_arrays[f"{anchor}_left_dual_transpose"],
            fiber_arrays[f"{anchor}_unstable_operator"],
        )
        seeds[anchor], _, seed_metrics[anchor] = _projected_transfer_seed(
            systems[anchor],
            generators[anchor],
            outputs[anchor],
            common_arrays[f"{anchor}_augmented_restriction"],
            common_arrays[f"{anchor}_augmented_lifting"],
            frequencies,
            heldout_frequencies,
            max(manifest.HIDDEN_ORDERS),
        )
        base_metrics[anchor]["passed"] = _base_pass(
            base_metrics[anchor], seed_metrics[anchor], gates
        )
        base_passed &= base_metrics[anchor]["passed"]
        prepared[anchor] = _prepare_reference(
            systems[anchor], frequencies, heldout_frequencies
        )
    candidate_metrics = []
    error_arrays = {}
    selected = None
    selected_arrays = None
    best = None
    best_arrays = None
    all_candidate_numerical = True
    for order in manifest.HIDDEN_ORDERS:
        item = {
            "hidden_order": order,
            "online_dimension": (
                manifest.PHYSICAL_DIMENSION
                + manifest.EXACT_NONSTABLE_DIMENSION
                + order
            ),
        }
        model_arrays = {}
        for anchor in ("primary", "heldout"):
            item[anchor], arrays = _candidate(
                systems[anchor],
                prepared[anchor],
                seeds[anchor],
                order,
                frequencies,
                heldout_frequencies,
                gates,
            )
            all_candidate_numerical &= item[anchor]["numerical_passed"]
            for name, value in arrays.items():
                if name.endswith("_errors"):
                    error_arrays[f"Z{order}_{anchor}_{name}"] = value
                else:
                    model_arrays[f"{anchor}_{name}"] = value
        primary_orthonormal = np.linalg.qr(
            model_arrays["primary_hidden_truth_trial"], mode="reduced"
        )[0]
        heldout_orthonormal = np.linalg.qr(
            model_arrays["heldout_hidden_truth_trial"], mode="reduced"
        )[0]
        cosines = np.linalg.svd(
            primary_orthonormal.T @ heldout_orthonormal,
            compute_uv=False,
        )
        item["cross_anchor_hidden_principal_cosine_min"] = float(
            np.min(cosines)
        )
        item["cross_anchor_hidden_largest_principal_angle_degrees"] = float(
            np.degrees(
                np.arccos(np.clip(np.min(cosines), -1.0, 1.0))
            )
        )
        item["cross_anchor_passed"] = bool(
            item["cross_anchor_hidden_principal_cosine_min"]
            >= gates["cross_anchor_hidden_principal_cosine_min"]
        )
        item["joint_passed"] = bool(
            base_passed
            and item["cross_anchor_passed"]
            and item["primary"]["passed"]
            and item["heldout"]["passed"]
        )
        item["maximum_gate_ratio"] = _candidate_score(item, gates)
        candidate_metrics.append(item)
        if best is None or item["maximum_gate_ratio"] < best["maximum_gate_ratio"]:
            best = item
            best_arrays = model_arrays
        if item["joint_passed"]:
            selected = item
            selected_arrays = model_arrays
            break
    elapsed = float(time.perf_counter() - began)
    numerical_passed = bool(
        base_passed
        and all_candidate_numerical
        and np.isfinite(elapsed)
        and elapsed
        <= 3600.0 * frozen["contract"]["execution_budget"]["maximum_wall_hours"]
    )
    passed = bool(numerical_passed and selected is not None)
    if not numerical_passed:
        classification = NUMERICAL_FAIL_CLASSIFICATION
        authorized_next = None
    elif passed:
        classification = PASS_CLASSIFICATION
        authorized_next = "definitions_only_cross_anchor_parametric_alignment_manifest"
    else:
        classification = CAP_FAIL_CLASSIFICATION
        authorized_next = None
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "metrics.json",
        {
            "base_metrics": base_metrics,
            "seed_metrics": seed_metrics,
            "candidate_metrics": candidate_metrics,
            "selected": selected,
            "best": best,
            "numerical_passed": numerical_passed,
            "wall_seconds": elapsed,
        },
    )
    np.savez_compressed(
        CANONICAL_DIRECTORY / "candidate_errors.npz",
        training_angular_frequencies_per_second=frequencies,
        heldout_angular_frequencies_per_second=heldout_frequencies,
        **error_arrays,
    )
    np.savez_compressed(
        CANONICAL_DIRECTORY / "decisive_model.npz",
        **(
            selected_arrays
            if selected_arrays is not None
            else (best_arrays or {})
        ),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "numerical_passed": numerical_passed,
        "base_architecture_passed": base_passed,
        "selected_hidden_order": (
            None if selected is None else selected["hidden_order"]
        ),
        "selected_online_dimension": (
            None if selected is None else selected["online_dimension"]
        ),
        "selected_maximum_gate_ratio": (
            None if selected is None else selected["maximum_gate_ratio"]
        ),
        "best_hidden_order": None if best is None else best["hidden_order"],
        "best_maximum_gate_ratio": (
            None if best is None else best["maximum_gate_ratio"]
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
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
            "prior_package_hashes": _checksums(manifest.PRIOR_DIRECTORY),
            "fiber_package_hashes": _checksums(manifest.FIBER_DIRECTORY),
            "common_basis_package_hashes": _checksums(
                manifest.COMMON_BASIS_DIRECTORY
            ),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        complete_tools.THIS_RUNNER,
        memory_tools.THIS_RUNNER,
        prior_tools.THIS_RUNNER,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    detail = (
        f"Selected hidden order `{selected['hidden_order']}` and online dimension "
        f"`{selected['online_dimension']}` with maximum gate ratio "
        f"`{selected['maximum_gate_ratio']:.6e}`."
        if selected is not None
        else (
            "No hidden order through 130 passed. The best order was "
            f"`{best['hidden_order'] if best else None}` with maximum gate ratio "
            f"`{best['maximum_gate_ratio'] if best else None}`."
        )
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Square-root transfer-seeded reduction audit WP10c9d6c7c3b5c4f25aa",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "This saved-generator audit preserved all 28 nonstable modes exactly and reduced only the strictly stable complement. The Lyapunov square root was used to form conservative trial/test bases without raw-P inversion.",
                "",
                detail,
                "",
                f"Authorized next artifact: `{authorized_next}`. No online integrator, predictive cycle, or reduced slow evolution is authorized.",
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
    arguments = parser.parse_args()
    if not arguments.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
