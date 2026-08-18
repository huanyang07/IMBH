#!/usr/bin/env python3
"""Execute the relative-Hermite resolvent POD saved-generator audit."""

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


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_relative_hermite_resolvent_manifest_wp10c9d6c7c3b5c4f25ad as manifest  # noqa: E402
import run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m as memory_tools  # noqa: E402
import run_causal_inner_square_root_transfer_seeded_audit_wp10c9d6c7c3b5c4f25aa as square_tools  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ae"
MANIFEST_COMMIT = "d1c773b1f96431f25c5edb002d7a32a7a1230b3f"
MANIFEST_PARENT = "cf94c51466682207e035d93a928dd0a3426b9c5a"
MANIFEST_TREE = "adb8c7a912408afb515d9b3230dd867d5acb65f9"

PASS_CLASSIFICATION = (
    "two_anchor_relative_Hermite_resolvent_reduction_passed_"
    "parametric_alignment_manifest_authorized"
)
CAP_FAIL_CLASSIFICATION = (
    "relative_Hermite_resolvent_reduction_failed_within_R320_"
    "tangential_residual_greedy_reassessment_required"
)
NUMERICAL_FAIL_CLASSIFICATION = (
    "relative_Hermite_resolvent_numerical_failure_stop"
)

ARTIFACT = (
    "causal_inner_relative_hermite_resolvent_audit_"
    "wp10c9d6c7c3b5c4f25ae"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_relative_hermite_resolvent_audit_"
    "wp10c9d6c7c3b5c4f25ae.py"
)
THIS_TEST = (
    "tests/test_causal_inner_relative_hermite_resolvent_audit_"
    "wp10c9d6c7c3b5c4f25ae.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_RELATIVE_HERMITE_RESOLVENT_"
    "AUDIT_WP10C9D6C7C3B5C4F25AE_2026-08-18.md"
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
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


def _pole_defect(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return a symmetric relative matching defect for two pole multisets."""
    if reference.size != candidate.size:
        return math.inf
    values = []
    for pole in reference:
        denominator = np.maximum(np.maximum(np.abs(pole), np.abs(candidate)), 1.0)
        values.append(float(np.min(np.abs(pole - candidate) / denominator)))
    for pole in candidate:
        denominator = np.maximum(np.maximum(np.abs(pole), np.abs(reference)), 1.0)
        values.append(float(np.min(np.abs(pole - reference) / denominator)))
    return float(max(values, default=0.0))


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("relative-Hermite manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("relative-Hermite manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("relative-Hermite manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["heldout_information_used_in_basis"]
        or contract["execution_budget"]["candidate_hidden_orders"] != list(manifest.HIDDEN_ORDERS)
        or contract["relative_Hermite_resolvent_POD"]["heldout_information_may_influence_basis"]
    ):
        raise RuntimeError("relative-Hermite execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(manifest.PARENT_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent input changed: {name}")
    for name, expected in contract["fiber_decisive_hashes"].items():
        if _sha(manifest.FIBER_DIRECTORY / name) != expected:
            raise RuntimeError(f"fiber input changed: {name}")
    saved_paths = {
        "primary_generator": manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz",
        "primary_output": manifest.PRIMARY_GENERATOR_DIRECTORY / "projection.npz",
        "heldout_generator_and_output": manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz",
        "R32_projection": manifest.R32_DIRECTORY / "R32_projection_promotion.npz",
        "frequency_ladder": manifest.R32_DIRECTORY / "R32_transfer.npz",
    }
    for name, path in saved_paths.items():
        if _sha(path) != contract["saved_input_hashes"][name]:
            raise RuntimeError(f"saved input changed: {name}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("relative-Hermite audit requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _frequency_half_widths(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.size < 2 or np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("training frequencies must be a strictly increasing vector")
    widths = np.empty_like(frequencies)
    widths[0] = 0.5 * (frequencies[1] - frequencies[0])
    widths[-1] = 0.5 * (frequencies[-1] - frequencies[-2])
    widths[1:-1] = 0.5 * np.minimum(
        frequencies[1:-1] - frequencies[:-2],
        frequencies[2:] - frequencies[1:-1],
    )
    return widths


def _relative_hermite_basis(
    operator: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
    frequencies: np.ndarray,
    physical_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    dimension = operator.shape[0]
    identity = np.eye(dimension)
    widths = _frequency_half_widths(frequencies)
    covariance = np.zeros((dimension, dimension))
    maximum_residual = 0.0
    forcing_norm = max(float(np.linalg.norm(forcing)), np.finfo(float).tiny)
    groups = 0
    for frequency, width in zip(frequencies, widths, strict=True):
        matrix = 1j * frequency * identity - operator
        state = np.linalg.solve(matrix, forcing)
        derivative = np.linalg.solve(matrix, -1j * state)
        maximum_residual = max(
            maximum_residual,
            float(np.linalg.norm(matrix @ state - forcing) / forcing_norm),
            float(
                np.linalg.norm(matrix @ derivative + 1j * state)
                / max(float(np.linalg.norm(state)), np.finfo(float).tiny)
            ),
        )
        for snapshot in (state, width * derivative):
            for row_slice in (slice(0, physical_rows), slice(physical_rows, None)):
                observed = observation[row_slice] @ snapshot
                scale = max(float(np.linalg.norm(observed)) ** 2, np.finfo(float).tiny)
                covariance += np.real(snapshot @ snapshot.conj().T) / scale
                groups += 1
    covariance /= float(groups)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    effective_rank = int(np.count_nonzero(eigenvalues > eigenvalues[0] * 1.0e-14))
    diagnostic_index = min(129, eigenvalues.size - 1)
    metrics = {
        "snapshot_covariance_effective_rank": effective_rank,
        "snapshot_covariance_largest_eigenvalue": float(eigenvalues[0]),
        "snapshot_covariance_eigenvalue_at_130": float(eigenvalues[diagnostic_index]),
        "snapshot_covariance_relative_eigenvalue_at_130": float(eigenvalues[diagnostic_index] / eigenvalues[0]),
        "maximum_snapshot_solve_relative_residual": maximum_residual,
        "snapshot_group_count": groups,
        "heldout_midpoint_responses_used": False,
        "shared_DC_training_control_used": bool(frequencies[0] == 0.0),
    }
    return eigenvectors, eigenvalues, metrics


def _prepare_reference(system: dict[str, np.ndarray], frequencies: np.ndarray, heldout: np.ndarray) -> dict:
    forcing, observation, direct, input_scales, output_scales = memory_tools._normalize_system(
        system["hidden_forcing"], system["combined_observation"], system["combined_direct"]
    )
    training_reference, training_residual = memory_tools._frequency_response(
        system["hidden_operator"], forcing, observation, direct, frequencies
    )
    heldout_reference, heldout_residual = memory_tools._frequency_response(
        system["hidden_operator"], forcing, observation, direct, heldout
    )
    return {
        "forcing": forcing,
        "observation": observation,
        "direct": direct,
        "input_scales": input_scales,
        "output_scales": output_scales,
        "training_reference": training_reference,
        "heldout_reference": heldout_reference,
        "reference_frequency_residual": max(training_residual, heldout_residual),
    }


def _candidate(
    system: dict[str, np.ndarray],
    prepared: dict,
    hidden_vectors: np.ndarray,
    order: int,
    frequencies: np.ndarray,
    heldout: np.ndarray,
    gates: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    q = manifest.PHYSICAL_DIMENSION
    hidden_trial = system["hidden_basis"] @ hidden_vectors[:, :order]
    trial = np.hstack((system["conservative_lift"], hidden_trial))
    test = np.hstack((system["conservative_map"].T, hidden_trial))
    gram = trial.T @ trial
    reduced = test.T @ system["whitened_operator"] @ trial
    hidden_operator = reduced[q:, q:]
    hidden_forcing = reduced[q:, :q] / prepared["input_scales"][None, :]
    observation_physical = np.vstack((reduced[:q, q:], system["whitened_output"] @ hidden_trial))
    direct_physical = np.vstack((reduced[:q, :q], system["whitened_output"] @ system["conservative_lift"]))
    observation = observation_physical / prepared["output_scales"][:, None]
    direct = direct_physical / prepared["output_scales"][:, None] / prepared["input_scales"][None, :]
    training, training_residual = memory_tools._frequency_response(hidden_operator, hidden_forcing, observation, direct, frequencies)
    heldout_response, heldout_residual = memory_tools._frequency_response(hidden_operator, hidden_forcing, observation, direct, heldout)
    blocks = {}; errors = {}
    for label, row_slice in (("resolved_self_energy", slice(0, q)), ("conservative_face_flux", slice(q, None))):
        training_metrics, training_errors = square_tools._block_metrics(training, prepared["training_reference"], prepared["direct"], row_slice)
        heldout_metrics, heldout_errors = square_tools._block_metrics(heldout_response, prepared["heldout_reference"], prepared["direct"], row_slice)
        blocks[label] = {
            **{f"training_{key}": value for key, value in training_metrics.items()},
            **{f"heldout_{key}": value for key, value in heldout_metrics.items()},
        }
        for prefix, values in (("training", training_errors), ("heldout", heldout_errors)):
            for name, value in values.items():
                errors[f"{label}_{prefix}_{name}"] = value
    lyapunov_target = trial.T @ (system["whitened_operator"] + system["whitened_operator"].T) @ trial
    lyapunov_residual = gram @ reduced + reduced.T @ gram - lyapunov_target
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
        "reduced_Lyapunov_identity_relative_defect": float(np.linalg.norm(lyapunov_residual) / max(float(np.linalg.norm(lyapunov_target)), np.finfo(float).tiny)),
        "reduced_stable_spectral_abscissa_per_second": float(np.max(np.real(reduced_poles))),
        "complete_nonstable_eigenvalue_count": int(complete_nonstable.size),
        "extra_nonstable_eigenvalue_count": int(np.sum(np.real(reduced_poles) >= threshold)),
        "exact_nonstable_pole_relative_defect": _pole_defect(unstable_poles, complete_nonstable),
        "maximum_frequency_solve_relative_residual": max(prepared["reference_frequency_residual"], training_residual, heldout_residual),
        "blocks": blocks,
    }
    metrics["numerical_passed"] = bool(
        metrics["conservative_test_identity_defect"] <= gates["conservative_lift_identity_defect_max"]
        and metrics["trial_test_biorthogonality_defect"] <= gates["trial_test_biorthogonality_defect_max"]
        and metrics["hidden_conservative_annihilation_defect"] <= gates["hidden_conservative_annihilation_defect_max"]
        and metrics["reduced_Lyapunov_identity_relative_defect"] <= gates["reduced_Lyapunov_identity_relative_defect_max"]
        and metrics["maximum_frequency_solve_relative_residual"] <= gates["maximum_frequency_solve_relative_residual_max"]
    )
    metrics["spectral_passed"] = bool(
        metrics["reduced_stable_spectral_abscissa_per_second"] <= gates["reduced_stable_spectral_abscissa_per_second_max"]
        and metrics["complete_nonstable_eigenvalue_count"] == gates["complete_nonstable_eigenvalue_count_equal"]
        and metrics["extra_nonstable_eigenvalue_count"] <= gates["extra_nonstable_eigenvalue_count_max"]
        and metrics["exact_nonstable_pole_relative_defect"] <= gates["exact_nonstable_pole_relative_defect_max"]
    )
    metrics["transfer_passed"] = bool(all(square_tools._block_pass(blocks[label], gates[label]) for label in blocks))
    metrics["passed"] = bool(metrics["numerical_passed"] and metrics["spectral_passed"] and metrics["transfer_passed"])
    return metrics, {
        "hidden_truth_trial": system["stable_basis"] @ system["inverse_square_root"] @ hidden_trial,
        "stable_reduced_operator": reduced,
        "whitened_trial": trial,
        "whitened_test": test,
        "metric": gram,
        "normalized_hidden_forcing": hidden_forcing,
        "normalized_combined_observation": observation,
        "normalized_combined_direct": direct,
        **errors,
    }


def _base_pass(base: dict, snapshot: dict, gates: dict) -> bool:
    return bool(
        base["stable_Lyapunov_relative_residual"] <= gates["stable_Lyapunov_relative_residual_max"]
        and base["square_root_reconstruction_relative_defect"] <= gates["square_root_reconstruction_relative_defect_max"]
        and base["whitened_Lyapunov_relative_defect"] <= gates["whitened_Lyapunov_relative_defect_max"]
        and base["conservative_lift_identity_defect"] <= gates["conservative_lift_identity_defect_max"]
        and base["full_trial_test_biorthogonality_defect"] <= gates["trial_test_biorthogonality_defect_max"]
        and snapshot["snapshot_covariance_effective_rank"] >= gates["snapshot_covariance_effective_rank_min"]
        and snapshot["maximum_snapshot_solve_relative_residual"] <= gates["maximum_snapshot_solve_relative_residual_max"]
    )


def _candidate_score(item: dict, gates: dict) -> float:
    cosine = item["cross_anchor_hidden_principal_cosine_min"]
    ratios = [gates["cross_anchor_hidden_principal_cosine_min"] / max(cosine, np.finfo(float).tiny)]
    for anchor in ("primary", "heldout"):
        metrics = item[anchor]
        ratios.extend((
            metrics["trial_test_biorthogonality_defect"] / gates["trial_test_biorthogonality_defect_max"],
            metrics["reduced_Lyapunov_identity_relative_defect"] / gates["reduced_Lyapunov_identity_relative_defect_max"],
        ))
        for block in ("resolved_self_energy", "conservative_face_flux"):
            for prefix in ("training", "heldout"):
                for name, limit in gates[block].items():
                    ratios.append(metrics["blocks"][block][f"{prefix}_{name.removesuffix('_max')}"] / limit)
    return float(max(ratios))


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": MANIFEST_COMMIT, "latest_work_package": WORK_PACKAGE})
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("relative-Hermite audit is already canonicalized")
    began = time.perf_counter()
    with np.load(manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz", allow_pickle=False) as source:
        primary_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    with np.load(manifest.PRIMARY_GENERATOR_DIRECTORY / "projection.npz", allow_pickle=False) as source:
        primary_output = np.asarray(source["output_map"], dtype=float)
    with np.load(manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz", allow_pickle=False) as source:
        heldout_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float); heldout_output = np.asarray(source["output_map"], dtype=float)
    with np.load(manifest.R32_DIRECTORY / "R32_projection_promotion.npz", allow_pickle=False) as source:
        restriction = np.asarray(source["resolved_restriction"], dtype=float)
    with np.load(manifest.R32_DIRECTORY / "R32_transfer.npz", allow_pickle=False) as source:
        frequencies = np.asarray(source["angular_frequencies_per_second"], dtype=float)
    heldout_frequencies = memory_tools._heldout_frequencies(frequencies)
    with np.load(manifest.FIBER_DIRECTORY / "decisive_fibers.npz", allow_pickle=False) as source:
        fiber = {name: np.asarray(source[name]) for name in source.files}
    generators = {"primary": primary_generator, "heldout": heldout_generator}; outputs = {"primary": primary_output, "heldout": heldout_output}
    gates = frozen["contract"]["binding_gates"]
    systems = {}; prepared = {}; hidden_vectors = {}; base_metrics = {}; snapshot_metrics = {}; base_passed = True
    for anchor in ("primary", "heldout"):
        systems[anchor], base_metrics[anchor] = square_tools._square_root_stable_system(
            generators[anchor], outputs[anchor], restriction,
            fiber[f"{anchor}_right_basis"], fiber[f"{anchor}_left_dual_transpose"], fiber[f"{anchor}_unstable_operator"],
        )
        prepared[anchor] = _prepare_reference(systems[anchor], frequencies, heldout_frequencies)
        hidden_vectors[anchor], _, snapshot_metrics[anchor] = _relative_hermite_basis(
            systems[anchor]["hidden_operator"], prepared[anchor]["forcing"], prepared[anchor]["observation"], frequencies, manifest.PHYSICAL_DIMENSION
        )
        base_metrics[anchor]["passed"] = _base_pass(base_metrics[anchor], snapshot_metrics[anchor], gates)
        base_passed &= base_metrics[anchor]["passed"]
    candidate_metrics = []; error_arrays = {}; selected = None; selected_arrays = None; best = None; best_arrays = None; all_candidate_numerical = True
    for order in manifest.HIDDEN_ORDERS:
        item = {"hidden_order": order, "online_dimension": manifest.PHYSICAL_DIMENSION + manifest.EXACT_NONSTABLE_DIMENSION + order}; model_arrays = {}
        for anchor in ("primary", "heldout"):
            item[anchor], arrays = _candidate(systems[anchor], prepared[anchor], hidden_vectors[anchor], order, frequencies, heldout_frequencies, gates)
            all_candidate_numerical &= item[anchor]["numerical_passed"]
            for name, value in arrays.items():
                if name.endswith("_errors"):
                    error_arrays[f"Z{order}_{anchor}_{name}"] = value
                else:
                    model_arrays[f"{anchor}_{name}"] = value
        primary_q = np.linalg.qr(model_arrays["primary_hidden_truth_trial"], mode="reduced")[0]
        heldout_q = np.linalg.qr(model_arrays["heldout_hidden_truth_trial"], mode="reduced")[0]
        cosines = np.linalg.svd(primary_q.T @ heldout_q, compute_uv=False)
        item["cross_anchor_hidden_principal_cosine_min"] = float(np.min(cosines))
        item["cross_anchor_hidden_largest_principal_angle_degrees"] = float(np.degrees(np.arccos(np.clip(np.min(cosines), -1.0, 1.0))))
        item["cross_anchor_passed"] = bool(item["cross_anchor_hidden_principal_cosine_min"] >= gates["cross_anchor_hidden_principal_cosine_min"])
        item["joint_passed"] = bool(base_passed and item["cross_anchor_passed"] and item["primary"]["passed"] and item["heldout"]["passed"])
        item["maximum_gate_ratio"] = _candidate_score(item, gates)
        candidate_metrics.append(item)
        if best is None or item["maximum_gate_ratio"] < best["maximum_gate_ratio"]:
            best = item; best_arrays = model_arrays
        if item["joint_passed"]:
            selected = item; selected_arrays = model_arrays; break
    elapsed = float(time.perf_counter() - began)
    numerical_passed = bool(base_passed and all_candidate_numerical and np.isfinite(elapsed) and elapsed <= 3600.0 * frozen["contract"]["execution_budget"]["maximum_wall_hours"])
    passed = bool(numerical_passed and selected is not None)
    if not numerical_passed:
        classification = NUMERICAL_FAIL_CLASSIFICATION; authorized_next = None
    elif passed:
        classification = PASS_CLASSIFICATION; authorized_next = "definitions_only_cross_anchor_parametric_alignment_manifest"
    else:
        classification = CAP_FAIL_CLASSIFICATION; authorized_next = None
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {"base_metrics": base_metrics, "snapshot_metrics": snapshot_metrics, "candidate_metrics": candidate_metrics, "selected": selected, "best": best, "numerical_passed": numerical_passed, "wall_seconds": elapsed})
    np.savez_compressed(CANONICAL_DIRECTORY / "candidate_errors.npz", training_angular_frequencies_per_second=frequencies, heldout_angular_frequencies_per_second=heldout_frequencies, **error_arrays)
    np.savez_compressed(CANONICAL_DIRECTORY / "decisive_model.npz", **(selected_arrays if selected_arrays is not None else (best_arrays or {})))
    summary = {
        "schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": classification, "passed": passed, "numerical_passed": numerical_passed,
        "base_architecture_passed": base_passed, "heldout_midpoint_information_used_in_basis": False, "shared_DC_training_control_used": True,
        "selected_hidden_order": None if selected is None else selected["hidden_order"], "selected_online_dimension": None if selected is None else selected["online_dimension"], "selected_maximum_gate_ratio": None if selected is None else selected["maximum_gate_ratio"],
        "best_hidden_order": None if best is None else best["hidden_order"], "best_maximum_gate_ratio": None if best is None else best["maximum_gate_ratio"],
        "new_nonlinear_roots": 0, "propagated_states": 0, "new_full_560_direction_generator_assemblies": 0, "new_truth_anchors": 0, "physical_failure_detected": False,
        "online_integrator_implementation_authorized": False, "predictive_cycle_authorized": False, "reduced_slow_evolution_authorized": False, "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {"manifest_commit": MANIFEST_COMMIT, "manifest_parent": MANIFEST_PARENT, "manifest_tree": MANIFEST_TREE, "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY), "parent_package_hashes": _checksums(manifest.PARENT_DIRECTORY), "fiber_package_hashes": _checksums(manifest.FIBER_DIRECTORY)})
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST, square_tools.THIS_RUNNER, memory_tools.THIS_RUNNER)
    _write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "CERTIFIED" if passed else "REJECTED", "execution_commit": _git("rev-parse", "HEAD"), "execution_tree": _git("rev-parse", "HEAD^{tree}"), "tracked_worktree_clean_at_start": True, "runner": THIS_RUNNER, "test": THIS_TEST, "report": REPORT_RELATIVE, "source_hashes": {relative: _sha(ROOT / relative) for relative in source_files}, "python": sys.version, "platform": platform.platform(), "thread_environment": THREAD_ENVIRONMENT})
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    detail = (f"Selected hidden order `{selected['hidden_order']}` and online dimension `{selected['online_dimension']}` with maximum gate ratio `{selected['maximum_gate_ratio']:.6e}`." if selected is not None else f"No hidden order through 130 passed. The best order was `{best['hidden_order'] if best else None}` with maximum gate ratio `{best['maximum_gate_ratio'] if best else None}`.")
    REPORT_PATH.write_text("\n".join((
        "# Relative-Hermite resolvent audit WP10c9d6c7c3b5c4f25ae", "", "## Classification", "", f"`{classification}`", "",
        "This saved-generator audit built each hidden basis directly from output-relative primal resolvent and frequency-derivative snapshots in the exact square-root conservative nullspace. Midpoint responses were used only for held-out evaluation; DC was the inherited shared training/control frequency.", "", detail, "",
        f"Authorized next artifact: `{authorized_next}`. No online integrator, predictive cycle, or reduced slow evolution is authorized.", "",
    )), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
