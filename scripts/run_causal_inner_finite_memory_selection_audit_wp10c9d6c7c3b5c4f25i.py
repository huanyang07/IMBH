#!/usr/bin/env python3
"""Execute the single-anchor stable finite-memory model-selection audit."""

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
from scipy.linalg import solve_continuous_lyapunov


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_finite_memory_selection_manifest_wp10c9d6c7c3b5c4f25h as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25i"
MANIFEST_COMMIT = "1d9e3b7aa9152aa837ee7a5ffad44aa8a8d3ed66"
MANIFEST_PARENT = "3231aa5fab0bd9ee7c980b6b0747d5019629f572"
MANIFEST_TREE = "47bd311142d910c87e801487434ecc761b221a46"

ARTIFACT = "causal_inner_finite_memory_selection_audit_wp10c9d6c7c3b5c4f25i"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_finite_memory_selection_audit_wp10c9d6c7c3b5c4f25i.py"
THIS_TEST = "tests/test_causal_inner_finite_memory_selection_audit_wp10c9d6c7c3b5c4f25i.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_FINITE_MEMORY_SELECTION_"
    "AUDIT_WP10C9D6C7C3B5C4F25I_2026-08-18.md"
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

PASS_CLASSIFICATION = "single_anchor_finite_memory_order_selected_cross_anchor_manifest_authorized"
COMPACT_FAIL_CLASSIFICATION = (
    "compact_finite_memory_failed_larger_conservative_coarse_PDE_fallback_required"
)
NUMERICAL_FAIL_CLASSIFICATION = "finite_memory_balancing_audit_failed_stop"


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
        raise RuntimeError("finite-memory manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("finite-memory manifest parent changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("finite-memory manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["balanced_truncation_screen_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["execution_budget"]["allowed_new_nonlinear_roots"] != 0
        or tuple(contract["balanced_truncation"]["candidate_memory_orders"])
        != manifest.MEMORY_ORDERS
    ):
        raise RuntimeError("finite-memory authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(manifest.PARENT_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent decisive array changed: {name}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("finite-memory execution requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _normalize_system(
    forcing: np.ndarray,
    observation: np.ndarray,
    direct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    forcing = np.asarray(forcing, dtype=float)
    observation = np.asarray(observation, dtype=float)
    direct = np.asarray(direct, dtype=float)
    input_scales = np.sqrt(np.sum(forcing * forcing, axis=0) + np.sum(direct * direct, axis=0))
    output_scales = np.sqrt(
        np.sum(observation * observation, axis=1) + np.sum(direct * direct, axis=1)
    )
    input_scales = np.where(input_scales > 0.0, input_scales, 1.0)
    output_scales = np.where(output_scales > 0.0, output_scales, 1.0)
    normalized_forcing = forcing / input_scales[None, :]
    normalized_observation = observation / output_scales[:, None]
    normalized_direct = direct / output_scales[:, None] / input_scales[None, :]
    return normalized_forcing, normalized_observation, normalized_direct, input_scales, output_scales


def _balanced_realization(
    operator: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    operator = np.asarray(operator, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    observation = np.asarray(observation, dtype=float)
    controllability_rhs = forcing @ forcing.T
    observability_rhs = observation.T @ observation
    controllability = solve_continuous_lyapunov(operator, -controllability_rhs)
    observability = solve_continuous_lyapunov(operator.T, -observability_rhs)
    controllability = 0.5 * (controllability + controllability.T)
    observability = 0.5 * (observability + observability.T)
    controllability_residual = (
        operator @ controllability + controllability @ operator.T + controllability_rhs
    )
    observability_residual = (
        operator.T @ observability + observability @ operator + observability_rhs
    )
    wc, uc = np.linalg.eigh(controllability)
    wo, uo = np.linalg.eigh(observability)
    wc_scale = max(float(np.max(np.abs(wc))), np.finfo(float).tiny)
    wo_scale = max(float(np.max(np.abs(wo))), np.finfo(float).tiny)
    wc_positive = wc > wc_scale * 1.0e-14
    wo_positive = wo > wo_scale * 1.0e-14
    controllability_factor = uc[:, wc_positive] * np.sqrt(wc[wc_positive])[None, :]
    observability_factor = uo[:, wo_positive] * np.sqrt(wo[wo_positive])[None, :]
    left, hankel, right_h = np.linalg.svd(
        observability_factor.T @ controllability_factor, full_matrices=False
    )
    metrics = {
        "controllability_gramian_relative_residual": float(
            np.linalg.norm(controllability_residual)
            / max(float(np.linalg.norm(controllability_rhs)), np.finfo(float).tiny)
        ),
        "observability_gramian_relative_residual": float(
            np.linalg.norm(observability_residual)
            / max(float(np.linalg.norm(observability_rhs)), np.finfo(float).tiny)
        ),
        "controllability_gramian_minimum_eigenvalue": float(np.min(wc)),
        "observability_gramian_minimum_eigenvalue": float(np.min(wo)),
        "controllability_numerical_rank": int(np.count_nonzero(wc_positive)),
        "observability_numerical_rank": int(np.count_nonzero(wo_positive)),
        "positive_hankel_singular_value_count": int(np.count_nonzero(hankel > 0.0)),
        "largest_hankel_singular_value": float(hankel[0]),
        "sixth_hankel_singular_value": float(hankel[5]) if hankel.size >= 6 else 0.0,
    }
    arrays = {
        "controllability_factor": controllability_factor,
        "observability_factor": observability_factor,
        "hankel_singular_values": hankel,
        "hankel_left_vectors": left,
        "hankel_right_vectors_transpose": right_h,
    }
    return arrays, metrics


def _truncate_balanced(
    operator: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
    balanced: dict[str, np.ndarray],
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if order == 0:
        return (
            np.empty((0, 0)),
            np.empty((0, forcing.shape[1])),
            np.empty((observation.shape[0], 0)),
            {"biorthogonality_defect": 0.0, "hankel_tail_bound": float(2.0 * np.sum(balanced["hankel_singular_values"]))},
        )
    hankel = balanced["hankel_singular_values"]
    if order > hankel.size or np.any(hankel[:order] <= 0.0):
        raise RuntimeError(f"memory order {order} exceeds positive balanced rank")
    rc = balanced["controllability_factor"]
    ro = balanced["observability_factor"]
    left = balanced["hankel_left_vectors"][:, :order]
    right = balanced["hankel_right_vectors_transpose"].T[:, :order]
    inverse_sqrt = 1.0 / np.sqrt(hankel[:order])
    trial = rc @ (right * inverse_sqrt[None, :])
    test = (inverse_sqrt[:, None] * left.T) @ ro.T
    reduced_operator = test @ operator @ trial
    reduced_forcing = test @ forcing
    reduced_observation = observation @ trial
    metrics = {
        "biorthogonality_defect": float(np.max(np.abs(test @ trial - np.eye(order)))),
        "hankel_tail_bound": float(2.0 * np.sum(hankel[order:])),
    }
    return reduced_operator, reduced_forcing, reduced_observation, metrics


def _frequency_response(
    operator: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
    direct: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    if operator.shape[0] == 0:
        return np.repeat(direct[None, :, :], frequencies.size, axis=0).astype(complex)
    identity = np.eye(operator.shape[0])
    return np.asarray(
        [
            direct + observation @ np.linalg.solve(1j * omega * identity - operator, forcing)
            for omega in frequencies
        ]
    )


def _candidate_metrics(
    order: int,
    operator: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
    direct: np.ndarray,
    frequencies: np.ndarray,
    reference: np.ndarray,
    balanced: dict[str, np.ndarray],
    gates: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    reduced_operator, reduced_forcing, reduced_observation, truncation = _truncate_balanced(
        operator, forcing, observation, balanced, order
    )
    approximation = _frequency_response(
        reduced_operator, reduced_forcing, reduced_observation, direct, frequencies
    )
    reference_dynamic = reference - direct[None, :, :]
    approximation_dynamic = approximation - direct[None, :, :]
    dynamic_errors = np.asarray(
        [
            np.linalg.norm(approximation_dynamic[index] - reference_dynamic[index])
            / max(float(np.linalg.norm(reference_dynamic[index])), np.finfo(float).tiny)
            for index in range(frequencies.size)
        ]
    )
    total_errors = np.asarray(
        [
            np.linalg.norm(approximation[index] - reference[index])
            / max(float(np.linalg.norm(reference[index])), np.finfo(float).tiny)
            for index in range(frequencies.size)
        ]
    )
    if order:
        poles = np.linalg.eigvals(reduced_operator)
        spectral_abscissa = float(np.max(np.real(poles)))
        certificate = solve_continuous_lyapunov(
            reduced_operator.T, -np.eye(order)
        )
        certificate = 0.5 * (certificate + certificate.T)
        lyapunov_residual = (
            reduced_operator.T @ certificate
            + certificate @ reduced_operator
            + np.eye(order)
        )
        lyapunov_relative = float(np.linalg.norm(lyapunov_residual) / np.sqrt(order))
        certificate_minimum = float(np.min(np.linalg.eigvalsh(certificate)))
    else:
        poles = np.asarray((), dtype=complex)
        spectral_abscissa = -1.0e300
        lyapunov_relative = 0.0
        certificate_minimum = 1.0
    metrics = {
        "order": order,
        "online_continuous_dimension": manifest.BASE_RESOLVED_DIMENSION + order,
        "spectral_abscissa_per_second": spectral_abscissa,
        "lyapunov_dissipation_relative_residual": lyapunov_relative,
        "lyapunov_certificate_minimum_eigenvalue": certificate_minimum,
        "maximum_normalized_dynamic_transfer_relative_error": float(np.max(dynamic_errors)),
        "RMS_normalized_dynamic_transfer_relative_error": float(
            np.sqrt(np.mean(dynamic_errors * dynamic_errors))
        ),
        "DC_normalized_dynamic_transfer_relative_error": float(dynamic_errors[0]),
        "maximum_normalized_total_transfer_relative_error": float(np.max(total_errors)),
        **truncation,
    }
    metrics["passed"] = bool(
        metrics["spectral_abscissa_per_second"]
        <= gates["reduced_spectral_abscissa_per_second_max"]
        and metrics["lyapunov_dissipation_relative_residual"]
        <= gates["lyapunov_dissipation_residual_max"]
        and metrics["lyapunov_certificate_minimum_eigenvalue"]
        >= gates["lyapunov_certificate_minimum_eigenvalue_min"]
        and metrics["maximum_normalized_dynamic_transfer_relative_error"]
        <= gates["maximum_normalized_dynamic_transfer_relative_error_max"]
        and metrics["RMS_normalized_dynamic_transfer_relative_error"]
        <= gates["RMS_normalized_dynamic_transfer_relative_error_max"]
        and metrics["DC_normalized_dynamic_transfer_relative_error"]
        <= gates["DC_normalized_dynamic_transfer_relative_error_max"]
        and metrics["maximum_normalized_total_transfer_relative_error"]
        <= gates["maximum_normalized_total_transfer_relative_error_max"]
    )
    arrays = {
        "operator": reduced_operator,
        "forcing": reduced_forcing,
        "observation": reduced_observation,
        "poles": poles,
        "dynamic_errors": dynamic_errors,
        "total_errors": total_errors,
        "transfer": approximation,
    }
    return arrays, metrics


def _execute() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if SCRATCH_DIRECTORY.exists():
        raise RuntimeError("finite-memory scratch output already exists")
    began = time.perf_counter()
    with np.load(manifest.PARENT_DIRECTORY / "promotion.npz", allow_pickle=False) as source:
        operator = np.asarray(source["remaining_stable_operator"], dtype=float)
        forcing = np.asarray(source["remaining_stable_forcing"], dtype=float)
        observation = np.asarray(source["stable_observation"], dtype=float)
        direct = np.asarray(source["augmented_direct"], dtype=float)
    with np.load(manifest.PARENT_DIRECTORY / "transfer_real.npz", allow_pickle=False) as source:
        frequencies = np.asarray(source["angular_frequencies_per_second"], dtype=float)
        transfer_real = np.asarray(source["transfer_real"], dtype=float)
    with np.load(manifest.PARENT_DIRECTORY / "transfer_imag.npz", allow_pickle=False) as source:
        transfer_imag = np.asarray(source["transfer_imag"], dtype=float)
    reference_original = transfer_real + 1j * transfer_imag
    normalized_forcing, normalized_observation, normalized_direct, input_scales, output_scales = _normalize_system(
        forcing, observation, direct
    )
    reference = reference_original / output_scales[None, :, None] / input_scales[None, None, :]
    balanced, full_metrics = _balanced_realization(
        operator, normalized_forcing, normalized_observation
    )
    gates = frozen["contract"]["candidate_pass_requires"]
    model_arrays = {}
    candidate_metrics = []
    candidate_transfers = {}
    for order in manifest.MEMORY_ORDERS:
        arrays, metrics = _candidate_metrics(
            order,
            operator,
            normalized_forcing,
            normalized_observation,
            normalized_direct,
            frequencies,
            reference,
            balanced,
            gates,
        )
        candidate_metrics.append(metrics)
        model_arrays[f"operator_r{order}"] = arrays["operator"]
        model_arrays[f"forcing_r{order}"] = arrays["forcing"]
        model_arrays[f"observation_r{order}"] = arrays["observation"]
        model_arrays[f"pole_real_r{order}"] = np.real(arrays["poles"])
        model_arrays[f"pole_imag_r{order}"] = np.imag(arrays["poles"])
        model_arrays[f"dynamic_errors_r{order}"] = arrays["dynamic_errors"]
        model_arrays[f"total_errors_r{order}"] = arrays["total_errors"]
        candidate_transfers[f"transfer_real_r{order}"] = np.real(arrays["transfer"])
        candidate_transfers[f"transfer_imag_r{order}"] = np.imag(arrays["transfer"])
    selected = next((item["order"] for item in candidate_metrics if item["passed"]), None)
    full_gates = frozen["contract"]["full_order_numerical_pass_requires"]
    full_numerical_pass = bool(
        full_metrics["controllability_gramian_relative_residual"]
        <= full_gates["controllability_gramian_relative_residual_max"]
        and full_metrics["observability_gramian_relative_residual"]
        <= full_gates["observability_gramian_relative_residual_max"]
        and full_metrics["sixth_hankel_singular_value"] > 0.0
    )
    SCRATCH_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(
        SCRATCH_DIRECTORY / "balanced_basis.npz",
        input_scales=input_scales,
        output_scales=output_scales,
        hankel_singular_values=balanced["hankel_singular_values"],
    )
    _write_npz(
        SCRATCH_DIRECTORY / "candidate_models.npz",
        normalized_direct=normalized_direct,
        angular_frequencies_per_second=frequencies,
        **model_arrays,
    )
    _write_npz(
        SCRATCH_DIRECTORY / "candidate_transfers.npz",
        angular_frequencies_per_second=frequencies,
        reference_real=np.real(reference),
        reference_imag=np.imag(reference),
        **candidate_transfers,
    )
    roundtrip = True
    with np.load(SCRATCH_DIRECTORY / "candidate_models.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["normalized_direct"], normalized_direct)
        for order in manifest.MEMORY_ORDERS:
            roundtrip &= np.array_equal(source[f"operator_r{order}"], model_arrays[f"operator_r{order}"])
    with np.load(SCRATCH_DIRECTORY / "candidate_transfers.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["reference_real"], np.real(reference))
        roundtrip &= np.array_equal(source["reference_imag"], np.imag(reference))
    for item in candidate_metrics:
        item["database_roundtrip_bitwise"] = bool(roundtrip)
        item["passed"] = bool(item["passed"] and roundtrip)
    selected = next((item["order"] for item in candidate_metrics if item["passed"]), None)
    metrics = {
        "stage": "single_anchor_stable_balanced_memory_selection",
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_generator_assemblies": 0,
        "truth_anchors_queried": 0,
        "single_anchor_linear_memory_fits": len(manifest.MEMORY_ORDERS),
        "candidate_orders": manifest.MEMORY_ORDERS,
        "selected_order": selected,
        "selected_online_continuous_dimension": (
            manifest.BASE_RESOLVED_DIMENSION + selected if selected is not None else None
        ),
        "full_order_numerical_passed": full_numerical_pass,
        "database_roundtrip_bitwise": bool(roundtrip),
        "candidate_metrics": candidate_metrics,
        "production_memory_coefficients_authorized": False,
        "physical_failure_detected": False,
        "wall_seconds": float(time.perf_counter() - began),
        **full_metrics,
    }
    metrics["passed"] = bool(full_numerical_pass and selected is not None)
    _write_json(SCRATCH_DIRECTORY / "metrics.json", metrics)
    return metrics


def _classification(metrics: dict) -> tuple[str, str | None]:
    if not metrics["full_order_numerical_passed"]:
        return NUMERICAL_FAIL_CLASSIFICATION, None
    if metrics["selected_order"] is None:
        return COMPACT_FAIL_CLASSIFICATION, "definitions_only_larger_conservative_coarse_PDE_manifest"
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
        raise RuntimeError("finite-memory selection audit is already canonicalized")
    metrics = _read(SCRATCH_DIRECTORY / "metrics.json")
    classification, authorized_next = _classification(metrics)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": metrics["passed"],
        "full_order_numerical_passed": metrics["full_order_numerical_passed"],
        "candidate_orders": metrics["candidate_orders"],
        "selected_order": metrics["selected_order"],
        "selected_online_continuous_dimension": metrics[
            "selected_online_continuous_dimension"
        ],
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_generator_assemblies": 0,
        "truth_anchors_queried": 0,
        "single_anchor_linear_memory_fits": len(manifest.MEMORY_ORDERS),
        "production_memory_coefficients_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    for name in ("metrics.json", "balanced_basis.npz", "candidate_models.npz", "candidate_transfers.npz"):
        shutil.copy2(SCRATCH_DIRECTORY / name, CANONICAL_DIRECTORY / name)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_parent": MANIFEST_PARENT,
        "manifest_tree": MANIFEST_TREE,
        "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
        "promotion_package_hashes": _checksums(manifest.PARENT_DIRECTORY),
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
    candidate_lines = [
        (
            f"- r={item['order']}: pass={item['passed']}, max/RMS/DC dynamic error="
            f"{item['maximum_normalized_dynamic_transfer_relative_error']:.6e}/"
            f"{item['RMS_normalized_dynamic_transfer_relative_error']:.6e}/"
            f"{item['DC_normalized_dynamic_transfer_relative_error']:.6e}, "
            f"max total error={item['maximum_normalized_total_transfer_relative_error']:.6e}."
        )
        for item in metrics["candidate_metrics"]
    ]
    REPORT_PATH.write_text(
        "\n".join((
            "# Finite-memory selection audit WP10c9d6c7c3b5c4f25i",
            "",
            "## Classification",
            "",
            f"`{classification}`",
            "",
            "The hash-locked single-anchor R106/stable-454 system was used. No truth root, propagation, or generator assembly was performed.",
            "",
            *candidate_lines,
            "",
            f"Selected memory order: `{metrics['selected_order']}`; selected online continuous dimension: `{metrics['selected_online_continuous_dimension']}`. Full-order Gramian numerical pass: `{metrics['full_order_numerical_passed']}`.",
            "",
            f"Authorized next artifact: `{authorized_next}`. These coefficients remain single-anchor diagnostics; production coefficients, the online solver, a predictive cycle, and reduced slow evolution are not authorized.",
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
