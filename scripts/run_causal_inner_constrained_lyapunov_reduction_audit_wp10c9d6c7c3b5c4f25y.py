#!/usr/bin/env python3
"""Execute the conservative constrained-Lyapunov saved-generator audit."""

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
from scipy.linalg import null_space, solve_continuous_lyapunov


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_constrained_lyapunov_reduction_manifest_wp10c9d6c7c3b5c4f25x as manifest  # noqa: E402
import run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m as memory_tools  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25y"
MANIFEST_COMMIT = "06da43dd5334c9d86c88ba5c061480a346765b17"
MANIFEST_PARENT = "e91158055600e240c1d6ddb00f145aa00beac0a6"
MANIFEST_TREE = "560b3641eff1bdca9210fc501989df554b7ec17a"

PASS_CLASSIFICATION = (
    "two_anchor_conservative_constrained_lyapunov_reduction_passed_"
    "parametric_alignment_manifest_authorized"
)
CAP_FAIL_CLASSIFICATION = (
    "constrained_lyapunov_reduction_failed_within_R320_"
    "structured_basis_reassessment_required"
)
NUMERICAL_FAIL_CLASSIFICATION = (
    "constrained_lyapunov_reduction_numerical_failure_stop"
)

ARTIFACT = (
    "causal_inner_constrained_lyapunov_reduction_audit_"
    "wp10c9d6c7c3b5c4f25y"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_constrained_lyapunov_reduction_audit_"
    "wp10c9d6c7c3b5c4f25y.py"
)
THIS_TEST = (
    "tests/test_causal_inner_constrained_lyapunov_reduction_audit_"
    "wp10c9d6c7c3b5c4f25y.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CONSTRAINED_LYAPUNOV_REDUCTION_"
    "AUDIT_WP10C9D6C7C3B5C4F25Y_2026-08-18.md"
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
    if isinstance(value, bool): return value
    if isinstance(value, float): return value if math.isfinite(value) else None
    if isinstance(value, int): return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected: raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT: raise RuntimeError("constrained-Lyapunov manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT: raise RuntimeError("constrained-Lyapunov manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE: raise RuntimeError("constrained-Lyapunov manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"] or summary["authorized_next"] != WORK_PACKAGE
        or contract["execution_budget"]["candidate_hidden_orders"] != list(manifest.HIDDEN_ORDERS)
        or contract["execution_budget"]["allowed_new_full_560_direction_generator_assemblies"] != 0
    ): raise RuntimeError("constrained-Lyapunov execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected: raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(manifest.PARENT_DIRECTORY / name) != expected: raise RuntimeError(f"parent input changed: {name}")
    for name, expected in contract["fiber_decisive_hashes"].items():
        if _sha(manifest.FIBER_DIRECTORY / name) != expected: raise RuntimeError(f"fiber input changed: {name}")
    if require_clean and not _tracked_tree_clean(): raise RuntimeError("constrained-Lyapunov audit requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected: raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(float).tiny))


def _exact_stable_system(
    generator: np.ndarray,
    output: np.ndarray,
    restriction: np.ndarray,
    right_unstable: np.ndarray,
    left_unstable_transpose: np.ndarray,
    unstable_operator: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    stable_basis = null_space(left_unstable_transpose)
    stable_operator = stable_basis.T @ generator @ stable_basis
    conservative_map = restriction @ stable_basis
    stable_output = output @ stable_basis
    dimension = stable_operator.shape[0]
    certificate = solve_continuous_lyapunov(stable_operator.T, -np.eye(dimension))
    certificate = 0.5 * (certificate + certificate.T)
    certificate_residual = stable_operator.T @ certificate + certificate @ stable_operator + np.eye(dimension)
    inverse_P_Ct = np.linalg.solve(certificate, conservative_map.T)
    coordinate_metric = conservative_map @ inverse_P_Ct
    conservative_lift = np.linalg.solve(coordinate_metric.T, inverse_P_Ct.T).T
    hidden_basis = null_space(conservative_map)
    full_trial = np.hstack((conservative_lift, hidden_basis))
    full_gram = full_trial.T @ certificate @ full_trial
    full_test = certificate @ full_trial @ np.linalg.inv(full_gram)
    full_reduced = full_test.T @ stable_operator @ full_trial
    q = restriction.shape[0]
    hidden_operator = full_reduced[q:, q:]
    hidden_forcing = full_reduced[q:, :q]
    combined_observation = np.vstack((full_reduced[:q, q:], stable_output @ hidden_basis))
    combined_direct = np.vstack((full_reduced[:q, :q], stable_output @ conservative_lift))
    metrics = {
        "stable_Lyapunov_relative_residual": float(np.linalg.norm(certificate_residual) / np.sqrt(dimension)),
        "stable_Lyapunov_minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(certificate))),
        "stable_Lyapunov_condition_number": float(np.linalg.cond(certificate)),
        "conservative_map_rank": int(np.linalg.matrix_rank(conservative_map)),
        "conservative_lift_identity_defect": float(np.max(np.abs(conservative_map @ conservative_lift - np.eye(q)))),
        "full_conservative_test_identity_defect": float(np.max(np.abs(full_test[:, :q] - conservative_map.T))),
        "full_trial_test_biorthogonality_defect": float(np.max(np.abs(full_test.T @ full_trial - np.eye(dimension)))),
        "full_coordinate_reconstruction_relative_defect": _relative(full_trial @ full_test.T, np.eye(dimension)),
        "full_stable_spectral_abscissa_per_second": float(np.max(np.real(np.linalg.eigvals(full_reduced)))),
    }
    return {
        "stable_basis": stable_basis,
        "stable_operator": stable_operator,
        "stable_output": stable_output,
        "certificate": certificate,
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


def _empirical_balanced_trial(
    operator: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, dict]:
    n = operator.shape[0]
    identity = np.eye(n)
    controllability = np.zeros((n, n))
    observability = np.zeros((n, n))
    maximum_residual = 0.0
    forcing_norm = max(float(np.linalg.norm(forcing)), np.finfo(float).tiny)
    observation_norm = max(float(np.linalg.norm(observation)), np.finfo(float).tiny)
    for omega in frequencies:
        primal_matrix = 1j * omega * identity - operator
        primal = np.linalg.solve(primal_matrix, forcing)
        adjoint_matrix = -1j * omega * identity - operator.T
        adjoint = np.linalg.solve(adjoint_matrix, observation.T)
        maximum_residual = max(
            maximum_residual,
            float(np.linalg.norm(primal_matrix @ primal - forcing) / forcing_norm),
            float(np.linalg.norm(adjoint_matrix @ adjoint - observation.T) / observation_norm),
        )
        controllability += np.real(primal @ primal.conj().T)
        observability += np.real(adjoint @ adjoint.conj().T)
    controllability /= frequencies.size
    observability /= frequencies.size
    controllability = 0.5 * (controllability + controllability.T)
    observability = 0.5 * (observability + observability.T)
    wc, uc = np.linalg.eigh(controllability)
    wo, uo = np.linalg.eigh(observability)
    wc_scale = max(float(np.max(np.abs(wc))), np.finfo(float).tiny)
    wo_scale = max(float(np.max(np.abs(wo))), np.finfo(float).tiny)
    keep_c = wc > wc_scale * 1.0e-14
    keep_o = wo > wo_scale * 1.0e-14
    rc = uc[:, keep_c] * np.sqrt(wc[keep_c])[None, :]
    ro = uo[:, keep_o] * np.sqrt(wo[keep_o])[None, :]
    left, hankel, right_h = np.linalg.svd(ro.T @ rc, full_matrices=False)
    positive = hankel > hankel[0] * 1.0e-14
    right = right_h.T[:, positive]
    trial = rc @ (right / np.sqrt(hankel[positive])[None, :])
    return trial, {
        "empirical_controllability_rank": int(np.count_nonzero(keep_c)),
        "empirical_observability_rank": int(np.count_nonzero(keep_o)),
        "positive_empirical_hankel_rank": int(np.count_nonzero(positive)),
        "largest_empirical_hankel_singular_value": float(hankel[0]),
        "maximum_snapshot_solve_relative_residual": maximum_residual,
    }


def _prepare(system: dict[str, np.ndarray], frequencies: np.ndarray, heldout: np.ndarray) -> dict:
    forcing, observation, direct, input_scales, output_scales = memory_tools._normalize_system(
        system["hidden_forcing"], system["combined_observation"], system["combined_direct"]
    )
    training_reference, training_residual = memory_tools._frequency_response(system["hidden_operator"], forcing, observation, direct, frequencies)
    heldout_reference, heldout_residual = memory_tools._frequency_response(system["hidden_operator"], forcing, observation, direct, heldout)
    empirical_trial, empirical_metrics = _empirical_balanced_trial(system["hidden_operator"], forcing, observation, frequencies)
    return {
        "forcing": forcing,
        "observation": observation,
        "direct": direct,
        "input_scales": input_scales,
        "output_scales": output_scales,
        "training_reference": training_reference,
        "heldout_reference": heldout_reference,
        "reference_frequency_residual": max(training_residual, heldout_residual),
        "empirical_trial": empirical_trial,
        "empirical_metrics": empirical_metrics,
    }


def _P_orthonormalize(values: np.ndarray, certificate: np.ndarray) -> np.ndarray:
    gram = values.T @ certificate @ values
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    if np.any(eigenvalues <= 0.0): raise RuntimeError("candidate P-Gram matrix is not positive definite")
    return values @ (eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]) @ eigenvectors.T


def _block_metrics(approximation: np.ndarray, reference: np.ndarray, direct: np.ndarray, row_slice: slice) -> tuple[dict, dict[str, np.ndarray]]:
    dynamic, total, metrics = memory_tools._error_metrics(approximation[:, row_slice], reference[:, row_slice], direct[row_slice])
    return metrics, {"dynamic_errors": dynamic, "total_errors": total}


def _block_pass(metrics: dict, gates: dict) -> bool:
    return bool(all(metrics[f"{prefix}_{name.removesuffix('_max')}"] <= limit for prefix in ("training", "heldout") for name, limit in gates.items()))


def _pole_defect(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.size != candidate.size: return math.inf
    values = []
    for pole in reference:
        values.append(float(np.min(np.abs(pole - candidate) / np.maximum(np.maximum(np.abs(pole), np.abs(candidate)), 1.0))))
    for pole in candidate:
        values.append(float(np.min(np.abs(pole - reference) / np.maximum(np.maximum(np.abs(pole), np.abs(reference)), 1.0))))
    return float(max(values, default=0.0))


def _candidate(
    system: dict[str, np.ndarray],
    prepared: dict,
    order: int,
    frequencies: np.ndarray,
    heldout: np.ndarray,
    gates: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    if prepared["empirical_trial"].shape[1] < order: raise RuntimeError(f"empirical rank below order {order}")
    raw_hidden = system["hidden_basis"] @ prepared["empirical_trial"][:, :order]
    hidden_trial = _P_orthonormalize(raw_hidden, system["certificate"])
    trial = np.hstack((system["conservative_lift"], hidden_trial))
    gram = trial.T @ system["certificate"] @ trial
    test = system["certificate"] @ trial @ np.linalg.inv(gram)
    reduced = test.T @ system["stable_operator"] @ trial
    q = manifest.PHYSICAL_DIMENSION
    hidden_operator = reduced[q:, q:]
    hidden_forcing = reduced[q:, :q] / prepared["input_scales"][None, :]
    observation_physical = np.vstack((reduced[:q, q:], system["stable_output"] @ hidden_trial))
    direct_physical = np.vstack((reduced[:q, :q], system["stable_output"] @ system["conservative_lift"]))
    observation = observation_physical / prepared["output_scales"][:, None]
    direct = direct_physical / prepared["output_scales"][:, None] / prepared["input_scales"][None, :]
    training, training_residual = memory_tools._frequency_response(hidden_operator, hidden_forcing, observation, direct, frequencies)
    heldout_response, heldout_residual = memory_tools._frequency_response(hidden_operator, hidden_forcing, observation, direct, heldout)
    blocks = {}; errors = {}
    for label, row_slice in (("resolved_self_energy", slice(0, q)), ("conservative_face_flux", slice(q, None))):
        train_metrics, train_errors = _block_metrics(training, prepared["training_reference"], prepared["direct"], row_slice)
        held_metrics, held_errors = _block_metrics(heldout_response, prepared["heldout_reference"], prepared["direct"], row_slice)
        blocks[label] = {**{f"training_{k}": v for k, v in train_metrics.items()}, **{f"heldout_{k}": v for k, v in held_metrics.items()}}
        for prefix, data in (("training", train_errors), ("heldout", held_errors)):
            for name, value in data.items(): errors[f"{label}_{prefix}_{name}"] = value
    lyapunov_residual = gram @ reduced + reduced.T @ gram + trial.T @ trial
    reduced_poles = np.linalg.eigvals(reduced)
    unstable_poles = np.linalg.eigvals(system["unstable_operator"])
    complete_poles = np.concatenate((unstable_poles, reduced_poles))
    threshold = manifest.NONSTABLE_THRESHOLD_PER_SECOND
    complete_nonstable = complete_poles[np.real(complete_poles) >= threshold]
    metrics = {
        "hidden_order": order,
        "conservative_test_identity_defect": float(np.max(np.abs(test[:, :q] - system["conservative_map"].T))),
        "trial_test_biorthogonality_defect": float(np.max(np.abs(test.T @ trial - np.eye(q + order)))),
        "hidden_conservative_annihilation_defect": float(np.max(np.abs(system["conservative_map"] @ hidden_trial))),
        "reduced_Lyapunov_identity_relative_defect": float(np.linalg.norm(lyapunov_residual) / max(float(np.linalg.norm(trial.T @ trial)), np.finfo(float).tiny)),
        "reduced_stable_spectral_abscissa_per_second": float(np.max(np.real(reduced_poles))),
        "complete_nonstable_eigenvalue_count": int(complete_nonstable.size),
        "extra_nonstable_eigenvalue_count": int(np.sum(np.real(reduced_poles) >= threshold)),
        "exact_nonstable_pole_relative_defect": _pole_defect(unstable_poles, complete_nonstable),
        "maximum_frequency_solve_relative_residual": max(prepared["reference_frequency_residual"], training_residual, heldout_residual),
        "blocks": blocks,
    }
    numerical_passed = bool(
        metrics["conservative_test_identity_defect"] <= gates["conservative_test_identity_defect_max"]
        and metrics["trial_test_biorthogonality_defect"] <= gates["trial_test_biorthogonality_defect_max"]
        and metrics["hidden_conservative_annihilation_defect"] <= gates["hidden_conservative_annihilation_defect_max"]
        and metrics["reduced_Lyapunov_identity_relative_defect"] <= gates["reduced_Lyapunov_identity_relative_defect_max"]
        and metrics["maximum_frequency_solve_relative_residual"] <= gates["maximum_frequency_solve_relative_residual_max"]
    )
    spectral_passed = bool(
        metrics["reduced_stable_spectral_abscissa_per_second"] <= gates["reduced_stable_spectral_abscissa_per_second_max"]
        and metrics["complete_nonstable_eigenvalue_count"] == gates["complete_nonstable_eigenvalue_count_equal"]
        and metrics["extra_nonstable_eigenvalue_count"] <= gates["extra_nonstable_eigenvalue_count_max"]
        and metrics["exact_nonstable_pole_relative_defect"] <= gates["exact_nonstable_pole_relative_defect_max"]
    )
    transfer_passed = bool(all(_block_pass(blocks[label], gates[label]) for label in blocks))
    metrics.update({"numerical_passed": numerical_passed, "spectral_passed": spectral_passed, "transfer_passed": transfer_passed, "passed": bool(numerical_passed and spectral_passed and transfer_passed)})
    return metrics, {
        "hidden_truth_trial": system["stable_basis"] @ hidden_trial,
        "stable_reduced_operator": reduced,
        "stable_trial": trial,
        "stable_test": test,
        "normalized_hidden_forcing": hidden_forcing,
        "normalized_combined_observation": observation,
        "normalized_combined_direct": direct,
        **errors,
    }


def _candidate_score(item: dict, gates: dict) -> float:
    ratios = [item["cross_anchor_hidden_principal_cosine_min"] and gates["cross_anchor_hidden_principal_cosine_min"] / item["cross_anchor_hidden_principal_cosine_min"]]
    for anchor in ("primary", "heldout"):
        metrics = item[anchor]
        ratios.extend([
            metrics["conservative_test_identity_defect"] / gates["conservative_test_identity_defect_max"],
            metrics["reduced_Lyapunov_identity_relative_defect"] / gates["reduced_Lyapunov_identity_relative_defect_max"],
        ])
        for block in ("resolved_self_energy", "conservative_face_flux"):
            for prefix in ("training", "heldout"):
                for name, limit in gates[block].items(): ratios.append(metrics["blocks"][block][f"{prefix}_{name.removesuffix('_max')}"] / limit)
    return float(max(ratios))


def _base_pass(metrics: dict, gates: dict) -> bool:
    return bool(
        metrics["stable_Lyapunov_relative_residual"] <= gates["stable_Lyapunov_relative_residual_max"]
        and metrics["stable_Lyapunov_minimum_eigenvalue"] >= gates["stable_Lyapunov_minimum_eigenvalue_min"]
        and metrics["stable_Lyapunov_condition_number"] <= gates["stable_Lyapunov_condition_number_max"]
        and metrics["conservative_map_rank"] == manifest.PHYSICAL_DIMENSION
        and metrics["conservative_lift_identity_defect"] <= gates["conservative_lift_identity_defect_max"]
        and metrics["full_conservative_test_identity_defect"] <= gates["conservative_test_identity_defect_max"]
        and metrics["full_trial_test_biorthogonality_defect"] <= gates["trial_test_biorthogonality_defect_max"]
        and metrics["full_coordinate_reconstruction_relative_defect"] <= 5.0e-9
        and metrics["full_stable_spectral_abscissa_per_second"] <= gates["reduced_stable_spectral_abscissa_per_second_max"]
    )


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": MANIFEST_COMMIT, "latest_work_package": WORK_PACKAGE})
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists(): raise RuntimeError("constrained-Lyapunov audit is already canonicalized")
    began = time.perf_counter()
    with np.load(manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz", allow_pickle=False) as source: primary_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    with np.load(manifest.PRIMARY_GENERATOR_DIRECTORY / "projection.npz", allow_pickle=False) as source: primary_output = np.asarray(source["output_map"], dtype=float)
    with np.load(manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz", allow_pickle=False) as source:
        heldout_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float); heldout_output = np.asarray(source["output_map"], dtype=float)
    with np.load(manifest.R32_DIRECTORY / "R32_projection_promotion.npz", allow_pickle=False) as source: restriction = np.asarray(source["resolved_restriction"], dtype=float)
    with np.load(manifest.R32_DIRECTORY / "R32_transfer.npz", allow_pickle=False) as source: frequencies = np.asarray(source["angular_frequencies_per_second"], dtype=float)
    heldout_frequencies = memory_tools._heldout_frequencies(frequencies)
    with np.load(manifest.FIBER_DIRECTORY / "decisive_fibers.npz", allow_pickle=False) as source:
        fiber_arrays = {name: np.asarray(source[name]) for name in source.files}
    generators = {"primary": primary_generator, "heldout": heldout_generator}; outputs = {"primary": primary_output, "heldout": heldout_output}
    systems = {}; base_metrics = {}; prepared = {}; base_passed = True
    gates = frozen["contract"]["binding_gates"]
    for anchor in ("primary", "heldout"):
        systems[anchor], base_metrics[anchor] = _exact_stable_system(
            generators[anchor], outputs[anchor], restriction,
            fiber_arrays[f"{anchor}_right_basis"], fiber_arrays[f"{anchor}_left_dual_transpose"], fiber_arrays[f"{anchor}_unstable_operator"],
        )
        base_metrics[anchor]["passed"] = _base_pass(base_metrics[anchor], gates)
        base_passed &= base_metrics[anchor]["passed"]
        prepared[anchor] = _prepare(systems[anchor], frequencies, heldout_frequencies)
        base_passed &= bool(prepared[anchor]["empirical_metrics"]["maximum_snapshot_solve_relative_residual"] <= gates["maximum_frequency_solve_relative_residual_max"])
    candidate_metrics = []; error_arrays = {}; selected = None; selected_arrays = None; best = None; best_arrays = None
    for order in manifest.HIDDEN_ORDERS:
        item = {"hidden_order": order, "online_dimension": manifest.PHYSICAL_DIMENSION + manifest.EXACT_NONSTABLE_DIMENSION + order}; model_arrays = {}
        for anchor in ("primary", "heldout"):
            item[anchor], arrays = _candidate(systems[anchor], prepared[anchor], order, frequencies, heldout_frequencies, gates)
            for name, value in arrays.items():
                if name.endswith("_errors"): error_arrays[f"Z{order}_{anchor}_{name}"] = value
                else: model_arrays[f"{anchor}_{name}"] = value
        primary_hidden_orthonormal = np.linalg.qr(
            model_arrays["primary_hidden_truth_trial"], mode="reduced"
        )[0]
        heldout_hidden_orthonormal = np.linalg.qr(
            model_arrays["heldout_hidden_truth_trial"], mode="reduced"
        )[0]
        cosines = np.linalg.svd(
            primary_hidden_orthonormal.T @ heldout_hidden_orthonormal,
            compute_uv=False,
        )
        item["cross_anchor_hidden_principal_cosine_min"] = float(np.min(cosines))
        item["cross_anchor_hidden_largest_principal_angle_degrees"] = float(np.degrees(np.arccos(np.clip(np.min(cosines), -1.0, 1.0))))
        item["cross_anchor_passed"] = bool(item["cross_anchor_hidden_principal_cosine_min"] >= gates["cross_anchor_hidden_principal_cosine_min"])
        item["joint_passed"] = bool(base_passed and item["cross_anchor_passed"] and item["primary"]["passed"] and item["heldout"]["passed"])
        item["maximum_gate_ratio"] = _candidate_score(item, gates)
        candidate_metrics.append(item)
        if best is None or item["maximum_gate_ratio"] < best["maximum_gate_ratio"]: best = item; best_arrays = model_arrays
        if item["joint_passed"]: selected = item; selected_arrays = model_arrays; break
    elapsed = float(time.perf_counter() - began)
    numerical_passed = bool(base_passed and np.isfinite(elapsed) and elapsed <= 3600.0 * frozen["contract"]["execution_budget"]["maximum_wall_hours"])
    passed = bool(numerical_passed and selected is not None)
    if not numerical_passed: classification = NUMERICAL_FAIL_CLASSIFICATION; authorized_next = None
    elif passed: classification = PASS_CLASSIFICATION; authorized_next = "definitions_only_cross_anchor_parametric_alignment_manifest"
    else: classification = CAP_FAIL_CLASSIFICATION; authorized_next = "definitions_only_constrained_lyapunov_basis_reassessment_manifest"
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {"base_metrics": base_metrics, "empirical_metrics": {a: prepared[a]["empirical_metrics"] for a in prepared}, "candidate_metrics": candidate_metrics, "selected": selected, "best": best, "numerical_passed": numerical_passed, "wall_seconds": elapsed})
    np.savez_compressed(CANONICAL_DIRECTORY / "candidate_errors.npz", training_angular_frequencies_per_second=frequencies, heldout_angular_frequencies_per_second=heldout_frequencies, **error_arrays)
    np.savez_compressed(CANONICAL_DIRECTORY / "decisive_model.npz", **(selected_arrays if selected_arrays is not None else (best_arrays or {})))
    summary = {
        "schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": classification, "passed": passed, "numerical_passed": numerical_passed,
        "base_architecture_passed": base_passed, "selected_hidden_order": None if selected is None else selected["hidden_order"], "selected_online_dimension": None if selected is None else selected["online_dimension"],
        "selected_maximum_gate_ratio": None if selected is None else selected["maximum_gate_ratio"], "best_hidden_order": None if best is None else best["hidden_order"], "best_maximum_gate_ratio": None if best is None else best["maximum_gate_ratio"],
        "new_nonlinear_roots": 0, "propagated_states": 0, "new_full_560_direction_generator_assemblies": 0, "new_truth_anchors": 0, "physical_failure_detected": False,
        "online_integrator_implementation_authorized": False, "predictive_cycle_authorized": False, "reduced_slow_evolution_authorized": False, "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {"manifest_commit": MANIFEST_COMMIT, "manifest_parent": MANIFEST_PARENT, "manifest_tree": MANIFEST_TREE, "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY), "parent_package_hashes": _checksums(manifest.PARENT_DIRECTORY), "fiber_package_hashes": _checksums(manifest.FIBER_DIRECTORY)})
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST)
    _write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "CERTIFIED" if passed else "REJECTED", "execution_commit": _git("rev-parse", "HEAD"), "execution_tree": _git("rev-parse", "HEAD^{tree}"), "tracked_worktree_clean_at_start": True, "runner": THIS_RUNNER, "test": THIS_TEST, "report": REPORT_RELATIVE, "source_hashes": {relative: _sha(ROOT / relative) for relative in source_files}, "python": sys.version, "platform": platform.platform(), "thread_environment": THREAD_ENVIRONMENT})
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    detail = (f"Selected hidden order `{selected['hidden_order']}` and online dimension `{selected['online_dimension']}` with maximum normalized gate ratio `{selected['maximum_gate_ratio']:.6e}`." if selected is not None else f"No hidden order through 130 passed. The best order was `{best['hidden_order'] if best else None}` with maximum normalized gate ratio `{best['maximum_gate_ratio'] if best else None}`.")
    REPORT_PATH.write_text("\n".join(("# Constrained-Lyapunov reduction audit WP10c9d6c7c3b5c4f25y", "", "## Classification", "", f"`{classification}`", "", "This saved-generator audit kept all 28 nonstable modes exact and reduced only the strictly stable complement with a P-weighted test basis whose first 162 columns are the conservative restriction transpose.", "", detail, "", f"Authorized next artifact: `{authorized_next}`. No online integrator, predictive cycle, or reduced slow evolution is authorized.", "")), encoding="utf-8")
    _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true"); arguments = parser.parse_args()
    if not arguments.run: raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
