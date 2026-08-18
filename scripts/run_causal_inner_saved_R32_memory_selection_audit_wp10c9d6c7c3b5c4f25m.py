#!/usr/bin/env python3
"""Select a stable saved-generator memory architecture for the R32 backbone."""

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

import run_causal_inner_reduced_architecture_reassessment_manifest_wp10c9d6c7c3b5c4f25l as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25m"
MANIFEST_COMMIT = "e59e1fb485cef4809bb751e4a93b9b6583b9b503"
MANIFEST_PARENT = "062adf7486df4e67e9c576135f166dd42168d7b8"
MANIFEST_TREE = "8b33e3e0a0e7252c927787f78eac536ddde8624a"

ARTIFACT = "causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m.py"
THIS_TEST = "tests/test_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SAVED_R32_MEMORY_SELECTION_"
    "AUDIT_WP10C9D6C7C3B5C4F25M_2026-08-18.md"
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

NUMERICAL_FAIL_CLASSIFICATION = "saved_R32_memory_selection_numerical_failure_stop"
NO_MODEL_CLASSIFICATION = "saved_R32_memory_architectures_failed_reconsider_resolved_variables"


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


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("architecture manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("architecture manifest parent changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("architecture manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["saved_R32_memory_selection_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["execution_budget"]["allowed_new_nonlinear_roots"] != 0
        or tuple(contract["candidate_families"]["global_balanced_controls"]["orders"])
        != manifest.GLOBAL_BALANCED_ORDERS
        or tuple(
            contract["candidate_families"]["coherent_spatial_channel_models"][
                "spatial_channel_ranks"
            ]
        )
        != manifest.COHERENT_CHANNEL_RANKS
    ):
        raise RuntimeError("saved-R32 memory-selection authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(manifest.PARENT_DIRECTORY / name) != expected:
            raise RuntimeError(f"saved R32 input changed: {name}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("saved-R32 memory selection requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _normalize_system(
    forcing: np.ndarray,
    observation: np.ndarray,
    direct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_scales = np.sqrt(np.sum(forcing * forcing, axis=0) + np.sum(direct * direct, axis=0))
    output_scales = np.sqrt(
        np.sum(observation * observation, axis=1) + np.sum(direct * direct, axis=1)
    )
    input_scales = np.where(input_scales > 0.0, input_scales, 1.0)
    output_scales = np.where(output_scales > 0.0, output_scales, 1.0)
    return (
        forcing / input_scales[None, :],
        observation / output_scales[:, None],
        direct / output_scales[:, None] / input_scales[None, :],
        input_scales,
        output_scales,
    )


def _balanced_realization(
    operator: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
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
    rc = uc[:, wc_positive] * np.sqrt(wc[wc_positive])[None, :]
    ro = uo[:, wo_positive] * np.sqrt(wo[wo_positive])[None, :]
    left, hankel, right_h = np.linalg.svd(ro.T @ rc, full_matrices=False)
    return {
        "controllability_factor": rc,
        "observability_factor": ro,
        "hankel_singular_values": hankel,
        "hankel_left_vectors": left,
        "hankel_right_vectors_transpose": right_h,
    }, {
        "controllability_gramian_relative_residual": float(
            np.linalg.norm(controllability_residual)
            / max(float(np.linalg.norm(controllability_rhs)), np.finfo(float).tiny)
        ),
        "observability_gramian_relative_residual": float(
            np.linalg.norm(observability_residual)
            / max(float(np.linalg.norm(observability_rhs)), np.finfo(float).tiny)
        ),
        "controllability_numerical_rank": int(np.count_nonzero(wc_positive)),
        "observability_numerical_rank": int(np.count_nonzero(wo_positive)),
        "positive_hankel_singular_value_count": int(np.count_nonzero(hankel > 0.0)),
        "largest_hankel_singular_value": float(hankel[0]),
    }


def _truncate_balanced(
    operator: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
    balanced: dict[str, np.ndarray],
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
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
    return (
        test @ operator @ trial,
        test @ forcing,
        observation @ trial,
        {
            "biorthogonality_defect": float(np.max(np.abs(test @ trial - np.eye(order)))),
            "hankel_tail_bound": float(2.0 * np.sum(hankel[order:])),
        },
    )


def _frequency_response(
    operator: np.ndarray,
    forcing: np.ndarray,
    observation: np.ndarray,
    direct: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, float]:
    identity = np.eye(operator.shape[0])
    responses = []
    maximum_residual = 0.0
    forcing_norm = max(float(np.linalg.norm(forcing)), np.finfo(float).tiny)
    for omega in frequencies:
        matrix = 1j * omega * identity - operator
        solved = np.linalg.solve(matrix, forcing)
        maximum_residual = max(
            maximum_residual,
            float(np.linalg.norm(matrix @ solved - forcing) / forcing_norm),
        )
        responses.append(direct + observation @ solved)
    return np.asarray(responses), maximum_residual


def _heldout_frequencies(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.size < 2 or frequencies[0] != 0.0:
        raise RuntimeError("parent frequency ladder changed")
    midpoints = np.empty(frequencies.size - 1)
    midpoints[0] = 0.5 * frequencies[1]
    midpoints[1:] = np.sqrt(frequencies[1:-1] * frequencies[2:])
    return np.concatenate(([0.0], midpoints))


def _coherent_channel_bases(
    reference: np.ndarray,
    direct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dynamic = reference - direct[None, :, :]
    output_snapshot = np.concatenate(
        tuple(np.real(item) for item in dynamic) + tuple(np.imag(item) for item in dynamic),
        axis=1,
    )
    input_snapshot = np.concatenate(
        tuple(np.real(item) for item in dynamic) + tuple(np.imag(item) for item in dynamic),
        axis=0,
    )
    output_left, output_singular, _ = np.linalg.svd(output_snapshot, full_matrices=False)
    _, input_singular, input_right_h = np.linalg.svd(input_snapshot, full_matrices=False)
    return output_left, input_right_h.T, output_singular, input_singular


def _project_channels(
    forcing: np.ndarray,
    observation: np.ndarray,
    output_basis: np.ndarray,
    input_basis: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    output = output_basis[:, :rank]
    input_ = input_basis[:, :rank]
    return (forcing @ input_) @ input_.T, output @ (output.T @ observation)


def _error_metrics(
    approximation: np.ndarray,
    reference: np.ndarray,
    direct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    reference_dynamic = reference - direct[None, :, :]
    approximation_dynamic = approximation - direct[None, :, :]
    dynamic = np.asarray([
        np.linalg.norm(approximation_dynamic[index] - reference_dynamic[index])
        / max(float(np.linalg.norm(reference_dynamic[index])), np.finfo(float).tiny)
        for index in range(reference.shape[0])
    ])
    total = np.asarray([
        np.linalg.norm(approximation[index] - reference[index])
        / max(float(np.linalg.norm(reference[index])), np.finfo(float).tiny)
        for index in range(reference.shape[0])
    ])
    return dynamic, total, {
        "maximum_normalized_dynamic_transfer_relative_error": float(np.max(dynamic)),
        "RMS_normalized_dynamic_transfer_relative_error": float(np.sqrt(np.mean(dynamic * dynamic))),
        "DC_normalized_dynamic_transfer_relative_error": float(dynamic[0]),
        "maximum_normalized_total_transfer_relative_error": float(np.max(total)),
        "RMS_normalized_total_transfer_relative_error": float(np.sqrt(np.mean(total * total))),
        "DC_normalized_total_transfer_relative_error": float(total[0]),
    }


def _stability_metrics(operator: np.ndarray) -> dict:
    order = operator.shape[0]
    poles = np.linalg.eigvals(operator)
    certificate = solve_continuous_lyapunov(operator.T, -np.eye(order))
    certificate = 0.5 * (certificate + certificate.T)
    residual = operator.T @ certificate + certificate @ operator + np.eye(order)
    return {
        "spectral_abscissa_per_second": float(np.max(np.real(poles))),
        "lyapunov_dissipation_relative_residual": float(np.linalg.norm(residual) / np.sqrt(order)),
        "lyapunov_certificate_minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(certificate))),
    }


def _gates_pass(metrics: dict, gates: dict, prefix: str) -> bool:
    return bool(
        metrics["spectral_abscissa_per_second"]
        <= gates["reduced_spectral_abscissa_per_second_max"]
        and metrics["lyapunov_dissipation_relative_residual"]
        <= gates["lyapunov_dissipation_residual_max"]
        and metrics["lyapunov_certificate_minimum_eigenvalue"]
        >= gates["lyapunov_certificate_minimum_eigenvalue_min"]
        and metrics[f"{prefix}_maximum_normalized_dynamic_transfer_relative_error"]
        <= gates["maximum_normalized_dynamic_transfer_relative_error_max"]
        and metrics[f"{prefix}_RMS_normalized_dynamic_transfer_relative_error"]
        <= gates["RMS_normalized_dynamic_transfer_relative_error_max"]
        and metrics[f"{prefix}_DC_normalized_dynamic_transfer_relative_error"]
        <= gates["DC_normalized_dynamic_transfer_relative_error_max"]
        and metrics[f"{prefix}_maximum_normalized_total_transfer_relative_error"]
        <= gates["maximum_normalized_total_transfer_relative_error_max"]
        and metrics[f"{prefix}_RMS_normalized_total_transfer_relative_error"]
        <= gates["RMS_normalized_total_transfer_relative_error_max"]
        and metrics[f"{prefix}_DC_normalized_total_transfer_relative_error"]
        <= gates["DC_normalized_total_transfer_relative_error_max"]
    )


def _energy_rank(singular_values: np.ndarray, fraction: float) -> int:
    energy = np.asarray(singular_values, dtype=float) ** 2
    if not np.any(energy > 0.0):
        return 0
    cumulative = np.cumsum(energy) / np.sum(energy)
    return int(np.searchsorted(cumulative, fraction, side="left") + 1)


def _cumulative_rank(values: np.ndarray, fraction: float) -> int:
    weights = np.asarray(values, dtype=float)
    if not np.any(weights > 0.0):
        return 0
    cumulative = np.cumsum(weights) / np.sum(weights)
    return int(np.searchsorted(cumulative, fraction, side="left") + 1)


def _execute() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if SCRATCH_DIRECTORY.exists():
        raise RuntimeError("saved-R32 memory-selection scratch output already exists")
    began = time.perf_counter()
    with np.load(manifest.PARENT_DIRECTORY / "R32_projection_promotion.npz", allow_pickle=False) as source:
        operator = np.asarray(source["remaining_stable_operator"], dtype=float)
        forcing = np.asarray(source["remaining_stable_forcing"], dtype=float)
        observation = np.asarray(source["stable_observation"], dtype=float)
        direct = np.asarray(source["augmented_direct"], dtype=float)
    with np.load(manifest.PARENT_DIRECTORY / "R32_transfer.npz", allow_pickle=False) as source:
        training_frequencies = np.asarray(source["angular_frequencies_per_second"], dtype=float)
        training_original = np.asarray(source["transfer_real"], dtype=float) + 1j * np.asarray(
            source["transfer_imag"], dtype=float
        )
    normalized_forcing, normalized_observation, normalized_direct, input_scales, output_scales = _normalize_system(
        forcing, observation, direct
    )
    training_reference = (
        training_original / output_scales[None, :, None] / input_scales[None, None, :]
    )
    heldout_frequencies = _heldout_frequencies(training_frequencies)
    heldout_reference, heldout_solve_residual = _frequency_response(
        operator,
        normalized_forcing,
        normalized_observation,
        normalized_direct,
        heldout_frequencies,
    )
    balanced, full_metrics = _balanced_realization(
        operator, normalized_forcing, normalized_observation
    )
    output_basis, input_basis, output_singular, input_singular = _coherent_channel_bases(
        training_reference, normalized_direct
    )
    gates = frozen["contract"]["normalization_and_validation"][
        "candidate_pass_requires_training_and_heldout"
    ]
    model_arrays: dict[str, np.ndarray] = {
        "normalized_direct": normalized_direct,
        "input_scales": input_scales,
        "output_scales": output_scales,
    }
    error_arrays: dict[str, np.ndarray] = {}
    candidate_metrics = []
    balanced_models = {}
    for order in manifest.GLOBAL_BALANCED_ORDERS:
        reduced_operator, reduced_forcing, reduced_observation, truncation = _truncate_balanced(
            operator,
            normalized_forcing,
            normalized_observation,
            balanced,
            order,
        )
        balanced_models[order] = (reduced_operator, reduced_forcing, reduced_observation, truncation)
        stability = _stability_metrics(reduced_operator)
        family_specs = [("global_balanced", None, reduced_forcing, reduced_observation)]
        for rank in manifest.COHERENT_CHANNEL_RANKS:
            projected_forcing, projected_observation = _project_channels(
                reduced_forcing,
                reduced_observation,
                output_basis,
                input_basis,
                rank,
            )
            family_specs.append(("coherent_channels", rank, projected_forcing, projected_observation))
        for family, rank, candidate_forcing, candidate_observation in family_specs:
            label = f"{family}_r{order}" if rank is None else f"{family}_s{rank}_r{order}"
            training_approximation, training_solve_residual = _frequency_response(
                reduced_operator,
                candidate_forcing,
                candidate_observation,
                normalized_direct,
                training_frequencies,
            )
            heldout_approximation, candidate_heldout_solve_residual = _frequency_response(
                reduced_operator,
                candidate_forcing,
                candidate_observation,
                normalized_direct,
                heldout_frequencies,
            )
            training_dynamic, training_total, training_errors = _error_metrics(
                training_approximation, training_reference, normalized_direct
            )
            heldout_dynamic, heldout_total, heldout_errors = _error_metrics(
                heldout_approximation, heldout_reference, normalized_direct
            )
            metrics = {
                "label": label,
                "family": family,
                "memory_order": order,
                "spatial_channel_rank": rank,
                "online_continuous_dimension": manifest.BASE_ONLINE_DIMENSION + order,
                "training_maximum_frequency_solve_relative_residual": training_solve_residual,
                "heldout_maximum_frequency_solve_relative_residual": candidate_heldout_solve_residual,
                **stability,
                **truncation,
                **{f"training_{key}": value for key, value in training_errors.items()},
                **{f"heldout_{key}": value for key, value in heldout_errors.items()},
            }
            metrics["training_passed"] = _gates_pass(metrics, gates, "training")
            metrics["heldout_passed"] = _gates_pass(metrics, gates, "heldout")
            metrics["passed"] = bool(metrics["training_passed"] and metrics["heldout_passed"])
            candidate_metrics.append(metrics)
            model_arrays[f"operator_{label}"] = reduced_operator
            model_arrays[f"forcing_{label}"] = candidate_forcing
            model_arrays[f"observation_{label}"] = candidate_observation
            model_arrays[f"pole_real_{label}"] = np.real(np.linalg.eigvals(reduced_operator))
            model_arrays[f"pole_imag_{label}"] = np.imag(np.linalg.eigvals(reduced_operator))
            error_arrays[f"training_dynamic_errors_{label}"] = training_dynamic
            error_arrays[f"training_total_errors_{label}"] = training_total
            error_arrays[f"heldout_dynamic_errors_{label}"] = heldout_dynamic
            error_arrays[f"heldout_total_errors_{label}"] = heldout_total
    full_gates = frozen["contract"]["normalization_and_validation"][
        "full_order_numerical_pass_requires"
    ]
    parent_metrics = _read(manifest.PARENT_DIRECTORY / "metrics.json")
    maximum_reference_residual = max(
        heldout_solve_residual,
        float(parent_metrics["maximum_frequency_solve_relative_residual"]),
    )
    full_numerical_passed = bool(
        full_metrics["controllability_gramian_relative_residual"]
        <= full_gates["controllability_gramian_relative_residual_max"]
        and full_metrics["observability_gramian_relative_residual"]
        <= full_gates["observability_gramian_relative_residual_max"]
        and full_metrics["positive_hankel_singular_value_count"]
        >= full_gates["minimum_positive_hankel_singular_value_count"]
        and maximum_reference_residual
        <= full_gates["maximum_reference_frequency_solve_relative_residual"]
    )
    preference = {"coherent_channels": 0, "global_balanced": 1}
    passing = sorted(
        (item for item in candidate_metrics if item["passed"]),
        key=lambda item: (
            item["memory_order"],
            preference[item["family"]],
            item["spatial_channel_rank"] or 10**6,
        ),
    )
    selected = passing[0] if passing and full_numerical_passed else None
    frequency_rank_99 = []
    dynamic_training = training_reference - normalized_direct[None, :, :]
    for item in dynamic_training:
        frequency_rank_99.append(_energy_rank(np.linalg.svd(item, compute_uv=False), 0.99))
    SCRATCH_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(
        SCRATCH_DIRECTORY / "balanced_structure.npz",
        hankel_singular_values=balanced["hankel_singular_values"],
        aggregate_output_singular_values=output_singular,
        aggregate_input_singular_values=input_singular,
        training_frequency_dynamic_rank_99=np.asarray(frequency_rank_99, dtype=int),
        output_channel_basis=output_basis,
        input_channel_basis=input_basis,
    )
    _write_npz(SCRATCH_DIRECTORY / "candidate_models.npz", **model_arrays)
    _write_npz(
        SCRATCH_DIRECTORY / "candidate_errors.npz",
        training_angular_frequencies_per_second=training_frequencies,
        heldout_angular_frequencies_per_second=heldout_frequencies,
        **error_arrays,
    )
    roundtrip = True
    with np.load(SCRATCH_DIRECTORY / "candidate_models.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["normalized_direct"], normalized_direct)
        for item in candidate_metrics:
            roundtrip &= np.array_equal(
                source[f"operator_{item['label']}"], balanced_models[item["memory_order"]][0]
            )
            roundtrip &= np.array_equal(
                source[f"forcing_{item['label']}"], model_arrays[f"forcing_{item['label']}"]
            )
            roundtrip &= np.array_equal(
                source[f"observation_{item['label']}"],
                model_arrays[f"observation_{item['label']}"],
            )
    with np.load(SCRATCH_DIRECTORY / "candidate_errors.npz", allow_pickle=False) as source:
        roundtrip &= np.array_equal(source["training_angular_frequencies_per_second"], training_frequencies)
        roundtrip &= np.array_equal(source["heldout_angular_frequencies_per_second"], heldout_frequencies)
        for item in candidate_metrics:
            label = item["label"]
            roundtrip &= np.array_equal(
                source[f"training_dynamic_errors_{label}"],
                error_arrays[f"training_dynamic_errors_{label}"],
            )
            roundtrip &= np.array_equal(
                source[f"heldout_dynamic_errors_{label}"],
                error_arrays[f"heldout_dynamic_errors_{label}"],
            )
    for item in candidate_metrics:
        item["database_roundtrip_bitwise"] = bool(roundtrip)
        item["passed"] = bool(item["passed"] and roundtrip)
    if selected is not None:
        selected = next(item for item in candidate_metrics if item["label"] == selected["label"])
        if not selected["passed"]:
            selected = None
    structure_metrics = {
        "aggregate_output_rank_90": _energy_rank(output_singular, 0.90),
        "aggregate_output_rank_95": _energy_rank(output_singular, 0.95),
        "aggregate_output_rank_99": _energy_rank(output_singular, 0.99),
        "aggregate_output_rank_999": _energy_rank(output_singular, 0.999),
        "aggregate_input_rank_90": _energy_rank(input_singular, 0.90),
        "aggregate_input_rank_95": _energy_rank(input_singular, 0.95),
        "aggregate_input_rank_99": _energy_rank(input_singular, 0.99),
        "aggregate_input_rank_999": _energy_rank(input_singular, 0.999),
        "frequency_dynamic_rank_99_min": int(np.min(frequency_rank_99)),
        "frequency_dynamic_rank_99_max": int(np.max(frequency_rank_99)),
        "hankel_cumulative_order_90": _cumulative_rank(balanced["hankel_singular_values"], 0.90),
        "hankel_cumulative_order_95": _cumulative_rank(balanced["hankel_singular_values"], 0.95),
        "hankel_cumulative_order_99": _cumulative_rank(balanced["hankel_singular_values"], 0.99),
        "hankel_cumulative_order_999": _cumulative_rank(balanced["hankel_singular_values"], 0.999),
    }
    metrics = {
        "stage": "saved_R32_stable_memory_architecture_selection",
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 0,
        "new_truth_anchors": 0,
        "saved_generator_memory_fits": len(candidate_metrics),
        "candidate_metrics": candidate_metrics,
        "selected_label": selected["label"] if selected else None,
        "selected_family": selected["family"] if selected else None,
        "selected_memory_order": selected["memory_order"] if selected else None,
        "selected_spatial_channel_rank": selected["spatial_channel_rank"] if selected else None,
        "selected_online_continuous_dimension": selected["online_continuous_dimension"] if selected else None,
        "full_order_numerical_passed": full_numerical_passed,
        "maximum_reference_frequency_solve_relative_residual": maximum_reference_residual,
        "database_roundtrip_bitwise": bool(roundtrip),
        "production_memory_coefficients_authorized": False,
        "physical_failure_detected": False,
        "wall_seconds": float(time.perf_counter() - began),
        **full_metrics,
        **structure_metrics,
    }
    metrics["passed"] = bool(full_numerical_passed and selected is not None and roundtrip)
    _write_json(SCRATCH_DIRECTORY / "metrics.json", metrics)
    return metrics


def _classification(metrics: dict) -> tuple[str, str | None]:
    if not metrics["full_order_numerical_passed"]:
        return NUMERICAL_FAIL_CLASSIFICATION, None
    if metrics["selected_label"] is None:
        return NO_MODEL_CLASSIFICATION, "definitions_only_resolved_variable_reassessment_manifest"
    family = metrics["selected_family"]
    order = metrics["selected_memory_order"]
    if family == "coherent_channels":
        rank = metrics["selected_spatial_channel_rank"]
        name = f"coherent_rank_{rank}_order_{order}"
    else:
        name = f"global_balanced_order_{order}"
    return (
        f"single_anchor_R32_{name}_selected_cross_anchor_preflight_authorized",
        "definitions_only_common_resolved_subspace_cross_anchor_preflight_manifest",
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
        raise RuntimeError("saved-R32 memory-selection audit is already canonicalized")
    metrics = _read(SCRATCH_DIRECTORY / "metrics.json")
    classification, authorized_next = _classification(metrics)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": metrics["passed"],
        "full_order_numerical_passed": metrics["full_order_numerical_passed"],
        "selected_label": metrics["selected_label"],
        "selected_family": metrics["selected_family"],
        "selected_memory_order": metrics["selected_memory_order"],
        "selected_spatial_channel_rank": metrics["selected_spatial_channel_rank"],
        "selected_online_continuous_dimension": metrics["selected_online_continuous_dimension"],
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 0,
        "new_truth_anchors": 0,
        "saved_generator_memory_fits": metrics["saved_generator_memory_fits"],
        "production_memory_coefficients_authorized": False,
        "common_cross_anchor_subspace_certified": False,
        "online_reduced_solver_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    for name in ("metrics.json", "balanced_structure.npz", "candidate_models.npz", "candidate_errors.npz"):
        shutil.copy2(SCRATCH_DIRECTORY / name, CANONICAL_DIRECTORY / name)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_parent": MANIFEST_PARENT,
        "manifest_tree": MANIFEST_TREE,
        "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
        "R32_parent_package_hashes": _checksums(manifest.PARENT_DIRECTORY),
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
    candidate_lines = []
    for item in metrics["candidate_metrics"]:
        candidate_lines.append(
            f"- `{item['label']}`: pass `{item['passed']}`; training max/RMS dynamic "
            f"`{item['training_maximum_normalized_dynamic_transfer_relative_error']:.6e}/"
            f"{item['training_RMS_normalized_dynamic_transfer_relative_error']:.6e}`; "
            f"held-out max/RMS dynamic `{item['heldout_maximum_normalized_dynamic_transfer_relative_error']:.6e}/"
            f"{item['heldout_RMS_normalized_dynamic_transfer_relative_error']:.6e}`."
        )
    REPORT_PATH.write_text(
        "\n".join((
            "# Saved-R32 memory-selection audit WP10c9d6c7c3b5c4f25m",
            "",
            "## Classification",
            "",
            f"`{classification}`",
            "",
            "The hash-locked R180/stable-380 saved system was used. No truth root, propagation, new generator assembly, or new truth anchor was executed.",
            "",
            *candidate_lines,
            "",
            f"Selected model: `{metrics['selected_label']}` with online continuous dimension `{metrics['selected_online_continuous_dimension']}`. Cumulative Hankel-value orders at 90/95/99/99.9 percent are `{metrics['hankel_cumulative_order_90']}/{metrics['hankel_cumulative_order_95']}/{metrics['hankel_cumulative_order_99']}/{metrics['hankel_cumulative_order_999']}`.",
            "",
            "The coherent rank-1/2/3 hypotheses are evaluated rather than inferred from per-frequency matrix rank. The selected single-anchor coefficients remain diagnostic only.",
            "",
            f"Authorized next artifact: `{authorized_next}`. Production coefficients, an online solver, a predictive cycle, and reduced slow evolution remain blocked.",
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
