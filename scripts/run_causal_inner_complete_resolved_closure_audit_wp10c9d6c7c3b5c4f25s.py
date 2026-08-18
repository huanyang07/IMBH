#!/usr/bin/env python3
"""Execute the complete resolved self-energy plus face-flux closure audit."""

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
from scipy.linalg import null_space


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_common_resolved_subspace_cross_anchor_preflight_wp10c9d6c7c3b5c4f25o as generator_parent  # noqa: E402
import run_causal_inner_complete_resolved_closure_manifest_wp10c9d6c7c3b5c4f25r as manifest  # noqa: E402
import run_causal_inner_rank_adaptive_common_memory_audit_wp10c9d6c7c3b5c4f25q as parent  # noqa: E402
import run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m as memory_tools  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25s"
MANIFEST_COMMIT = "ed895b82d338b87f9d65a19ae5f01fca9bcf12b1"
MANIFEST_PARENT = "b62737cab1f62056fcfa374d5a4bd7bbe39319b1"
MANIFEST_TREE = "c07343fd26e34b6d9ef99eaf6f0a1da337a7ecc6"

ARTIFACT = (
    "causal_inner_complete_resolved_closure_audit_"
    "wp10c9d6c7c3b5c4f25s"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_complete_resolved_closure_audit_"
    "wp10c9d6c7c3b5c4f25s.py"
)
THIS_TEST = (
    "tests/test_causal_inner_complete_resolved_closure_audit_"
    "wp10c9d6c7c3b5c4f25s.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_COMPLETE_RESOLVED_CLOSURE_"
    "AUDIT_WP10C9D6C7C3B5C4F25S_2026-08-18.md"
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

PASS_CLASSIFICATION = (
    "two_anchor_complete_R196_memory_closure_passed_"
    "bounded_online_prototype_manifest_authorized"
)
CAP_FAIL_CLASSIFICATION = (
    "complete_R196_memory_closure_failed_within_R320_"
    "structured_closure_reassessment_required"
)
NUMERICAL_FAIL_CLASSIFICATION = (
    "complete_resolved_closure_numerical_failure_stop"
)


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
        raise RuntimeError("complete-closure manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("complete-closure manifest parent changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("complete-closure manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["resolved_dimension"] != manifest.RESOLVED_DIMENSION
        or contract["execution_budget"]["candidate_memory_orders"]
        != list(manifest.MEMORY_ORDERS)
        or contract["execution_budget"][
            "allowed_new_full_560_direction_generator_assemblies"
        ]
        != 0
    ):
        raise RuntimeError("complete-closure execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(parent.CANONICAL_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent decisive input changed: {name}")
    for name, expected in contract["saved_generator_hashes"].items():
        if _sha(generator_parent.CANONICAL_DIRECTORY / name) != expected:
            raise RuntimeError(f"saved generator input changed: {name}")
    _checksums(parent.CANONICAL_DIRECTORY)
    _checksums(generator_parent.CANONICAL_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("complete-closure audit requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), np.finfo(float).tiny)
    )


def _complete_blocks(
    generator: np.ndarray,
    restriction: np.ndarray,
    lifting: np.ndarray,
    output: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    stable_basis = null_space(restriction)
    resolved_direct = restriction @ generator @ lifting
    resolved_observation = restriction @ generator @ stable_basis
    stable_forcing = stable_basis.T @ generator @ lifting
    stable_operator = stable_basis.T @ generator @ stable_basis
    face_observation = output @ stable_basis
    face_direct = output @ lifting
    reconstructed = (
        lifting @ resolved_direct @ restriction
        + lifting @ resolved_observation @ stable_basis.T
        + stable_basis @ stable_forcing @ restriction
        + stable_basis @ stable_operator @ stable_basis.T
    )
    metrics = {
        "restriction_lifting_identity_defect": float(
            np.max(np.abs(restriction @ lifting - np.eye(restriction.shape[0])))
        ),
        "restriction_stable_annihilation_defect": float(
            np.max(np.abs(restriction @ stable_basis))
        ),
        "stable_lifting_annihilation_defect": float(
            np.max(np.abs(stable_basis.T @ lifting))
        ),
        "stable_basis_orthogonality_defect": float(
            np.max(
                np.abs(
                    stable_basis.T @ stable_basis
                    - np.eye(stable_basis.shape[1])
                )
            )
        ),
        "coordinate_reconstruction_relative_defect": _relative(
            reconstructed, generator
        ),
        "stable_spectral_abscissa_per_second": float(
            np.max(np.real(np.linalg.eigvals(stable_operator)))
        ),
    }
    return {
        "stable_basis": stable_basis,
        "resolved_direct": resolved_direct,
        "resolved_observation": resolved_observation,
        "stable_forcing": stable_forcing,
        "stable_operator": stable_operator,
        "face_observation": face_observation,
        "face_direct": face_direct,
    }, metrics


def _prepare_memory(
    blocks: dict[str, np.ndarray],
    frequencies: np.ndarray,
    heldout_frequencies: np.ndarray,
) -> dict:
    observation = np.vstack(
        (blocks["resolved_observation"], blocks["face_observation"])
    )
    direct = np.vstack((blocks["resolved_direct"], blocks["face_direct"]))
    forcing, observation, direct, input_scales, output_scales = (
        memory_tools._normalize_system(
            blocks["stable_forcing"], observation, direct
        )
    )
    training_reference, training_reference_residual = (
        memory_tools._frequency_response(
            blocks["stable_operator"],
            forcing,
            observation,
            direct,
            frequencies,
        )
    )
    heldout_reference, heldout_reference_residual = (
        memory_tools._frequency_response(
            blocks["stable_operator"],
            forcing,
            observation,
            direct,
            heldout_frequencies,
        )
    )
    balanced, full_metrics = memory_tools._balanced_realization(
        blocks["stable_operator"], forcing, observation
    )
    return {
        "operator": blocks["stable_operator"],
        "forcing": forcing,
        "observation": observation,
        "direct": direct,
        "input_scales": input_scales,
        "output_scales": output_scales,
        "training_reference": training_reference,
        "heldout_reference": heldout_reference,
        "reference_frequency_residual": max(
            training_reference_residual, heldout_reference_residual
        ),
        "balanced": balanced,
        "full_metrics": full_metrics,
    }


def _block_metrics(
    approximation: np.ndarray,
    reference: np.ndarray,
    direct: np.ndarray,
    row_slice: slice,
) -> tuple[dict, dict[str, np.ndarray]]:
    dynamic, total, metrics = memory_tools._error_metrics(
        approximation[:, row_slice],
        reference[:, row_slice],
        direct[row_slice],
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


def _spectral_metrics(
    exact_generator: np.ndarray,
    reduced_operator: np.ndarray,
    reduced_forcing: np.ndarray,
    reduced_observation: np.ndarray,
    reduced_direct: np.ndarray,
    input_scales: np.ndarray,
    output_scales: np.ndarray,
    threshold: float,
) -> dict:
    resolved = manifest.RESOLVED_DIMENSION
    physical_direct = (
        output_scales[:, None]
        * reduced_direct
        * input_scales[None, :]
    )
    physical_observation = output_scales[:, None] * reduced_observation
    physical_forcing = reduced_forcing * input_scales[None, :]
    closed = np.block(
        [
            [
                physical_direct[:resolved],
                physical_observation[:resolved],
            ],
            [physical_forcing, reduced_operator],
        ]
    )
    exact_poles = np.linalg.eigvals(exact_generator)
    reduced_poles = np.linalg.eigvals(closed)
    exact_nonstable = exact_poles[np.real(exact_poles) >= threshold]
    reduced_nonstable = reduced_poles[np.real(reduced_poles) >= threshold]

    def directed(left: np.ndarray, right: np.ndarray) -> float:
        if not left.size or not right.size:
            return math.inf if left.size != right.size else 0.0
        values = []
        for pole in left:
            distances = np.abs(pole - right) / np.maximum(
                np.maximum(np.abs(pole), np.abs(right)), 1.0
            )
            values.append(float(np.min(distances)))
        return float(max(values, default=0.0))

    defect = max(
        directed(exact_nonstable, reduced_nonstable),
        directed(reduced_nonstable, exact_nonstable),
    )
    return {
        "exact_nonstable_eigenvalue_count": int(exact_nonstable.size),
        "reduced_nonstable_eigenvalue_count": int(reduced_nonstable.size),
        "bidirectional_nearest_nonstable_eigenvalue_relative_defect": defect,
        "reduced_closed_spectral_abscissa_per_second": float(
            np.max(np.real(reduced_poles))
        ),
        "closed_generator": closed,
        "exact_nonstable_poles": exact_nonstable,
        "reduced_nonstable_poles": reduced_nonstable,
    }


def _candidate(
    prepared: dict,
    exact_generator: np.ndarray,
    frequencies: np.ndarray,
    heldout_frequencies: np.ndarray,
    order: int,
    transfer_gates: dict,
    numerical_gates: dict,
    spectral_gates: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    operator, forcing, observation, truncation = memory_tools._truncate_balanced(
        prepared["operator"],
        prepared["forcing"],
        prepared["observation"],
        prepared["balanced"],
        order,
    )
    training_approximation, training_residual = memory_tools._frequency_response(
        operator,
        forcing,
        observation,
        prepared["direct"],
        frequencies,
    )
    heldout_approximation, heldout_residual = memory_tools._frequency_response(
        operator,
        forcing,
        observation,
        prepared["direct"],
        heldout_frequencies,
    )
    split = manifest.RESOLVED_DIMENSION
    blocks = {}
    errors = {}
    for label, row_slice in (
        ("resolved_self_energy", slice(0, split)),
        ("conservative_face_flux", slice(split, None)),
    ):
        training_metrics, training_errors = _block_metrics(
            training_approximation,
            prepared["training_reference"],
            prepared["direct"],
            row_slice,
        )
        heldout_metrics, heldout_errors = _block_metrics(
            heldout_approximation,
            prepared["heldout_reference"],
            prepared["direct"],
            row_slice,
        )
        blocks[label] = {
            **{f"training_{key}": value for key, value in training_metrics.items()},
            **{f"heldout_{key}": value for key, value in heldout_metrics.items()},
        }
        for frequency_label, values in (
            ("training", training_errors),
            ("heldout", heldout_errors),
        ):
            for name, value in values.items():
                errors[f"{label}_{frequency_label}_{name}"] = value
    stability = memory_tools._stability_metrics(operator)
    spectral = _spectral_metrics(
        exact_generator,
        operator,
        forcing,
        observation,
        prepared["direct"],
        prepared["input_scales"],
        prepared["output_scales"],
        spectral_gates["nonstable_threshold_per_second"],
    )
    maximum_frequency_residual = max(
        prepared["reference_frequency_residual"],
        training_residual,
        heldout_residual,
    )
    numerical_passed = bool(
        prepared["full_metrics"]["controllability_gramian_relative_residual"]
        <= numerical_gates["controllability_gramian_relative_residual_max"]
        and prepared["full_metrics"]["observability_gramian_relative_residual"]
        <= numerical_gates["observability_gramian_relative_residual_max"]
        and truncation["biorthogonality_defect"]
        <= numerical_gates["biorthogonality_defect_max"]
        and maximum_frequency_residual
        <= numerical_gates["maximum_frequency_solve_relative_residual_max"]
        and stability["spectral_abscissa_per_second"]
        <= numerical_gates["memory_spectral_abscissa_per_second_max"]
        and stability["lyapunov_dissipation_relative_residual"]
        <= numerical_gates["lyapunov_dissipation_relative_residual_max"]
        and stability["lyapunov_certificate_minimum_eigenvalue"]
        >= numerical_gates["lyapunov_certificate_minimum_eigenvalue_min"]
    )
    spectral_passed = bool(
        spectral["exact_nonstable_eigenvalue_count"]
        == spectral["reduced_nonstable_eigenvalue_count"]
        and spectral[
            "bidirectional_nearest_nonstable_eigenvalue_relative_defect"
        ]
        <= spectral_gates["bidirectional_nearest_eigenvalue_relative_defect_max"]
    )
    transfer_passed = bool(
        all(
            _block_pass(blocks[label], transfer_gates[label])
            for label in blocks
        )
    )
    metrics = {
        "memory_order": order,
        **prepared["full_metrics"],
        **truncation,
        **stability,
        "maximum_frequency_solve_relative_residual": maximum_frequency_residual,
        "blocks": blocks,
        "exact_nonstable_eigenvalue_count": spectral[
            "exact_nonstable_eigenvalue_count"
        ],
        "reduced_nonstable_eigenvalue_count": spectral[
            "reduced_nonstable_eigenvalue_count"
        ],
        "bidirectional_nearest_nonstable_eigenvalue_relative_defect": spectral[
            "bidirectional_nearest_nonstable_eigenvalue_relative_defect"
        ],
        "reduced_closed_spectral_abscissa_per_second": spectral[
            "reduced_closed_spectral_abscissa_per_second"
        ],
        "numerical_passed": numerical_passed,
        "spectral_passed": spectral_passed,
        "transfer_passed": transfer_passed,
        "passed": bool(numerical_passed and spectral_passed and transfer_passed),
    }
    arrays = {
        "reduced_operator": operator,
        "reduced_forcing": forcing,
        "reduced_observation": observation,
        "normalized_direct": prepared["direct"],
        "input_scales": prepared["input_scales"],
        "output_scales": prepared["output_scales"],
        "closed_generator": spectral["closed_generator"],
        "exact_nonstable_poles": spectral["exact_nonstable_poles"],
        "reduced_nonstable_poles": spectral["reduced_nonstable_poles"],
        **errors,
    }
    return metrics, arrays


def _candidate_score(metrics: dict, transfer_gates: dict, spectral_limit: float) -> float:
    ratios = [
        metrics["bidirectional_nearest_nonstable_eigenvalue_relative_defect"]
        / spectral_limit
    ]
    for block, gates in transfer_gates.items():
        for prefix in ("training", "heldout"):
            for name, maximum in gates.items():
                ratios.append(
                    metrics["blocks"][block][
                        f"{prefix}_{name.removesuffix('_max')}"
                    ]
                    / maximum
                )
    return float(max(ratios))


def _classification(
    numerical_passed: bool, selected: dict | None
) -> tuple[str, str | None, bool]:
    if not numerical_passed:
        return NUMERICAL_FAIL_CLASSIFICATION, None, False
    if selected is not None:
        return (
            PASS_CLASSIFICATION,
            "definitions_only_bounded_R196_memory_online_integrator_manifest",
            True,
        )
    return (
        CAP_FAIL_CLASSIFICATION,
        "definitions_only_structured_resolved_closure_reassessment_manifest",
        False,
    )


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
                    "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
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
        raise RuntimeError("complete-closure audit is already canonicalized")
    began = time.perf_counter()
    with np.load(
        parent.CANONICAL_DIRECTORY / "decisive_basis.npz", allow_pickle=False
    ) as source:
        basis = {
            anchor: {
                "restriction": np.asarray(
                    source[f"{anchor}_augmented_restriction"], dtype=float
                ),
                "lifting": np.asarray(
                    source[f"{anchor}_augmented_lifting"], dtype=float
                ),
            }
            for anchor in ("primary", "heldout")
        }
    with np.load(
        generator_parent.manifest.GENERATOR_DIRECTORY / "descriptor_A.npz",
        allow_pickle=False,
    ) as source:
        primary_generator = np.asarray(
            source["complete_fixed_Q_generator"], dtype=float
        )
    with np.load(
        generator_parent.manifest.GENERATOR_DIRECTORY / "projection.npz",
        allow_pickle=False,
    ) as source:
        primary_output = np.asarray(source["output_map"], dtype=float)
    with np.load(
        generator_parent.CANONICAL_DIRECTORY / "heldout_generator.npz",
        allow_pickle=False,
    ) as source:
        heldout_generator = np.asarray(
            source["complete_fixed_Q_generator"], dtype=float
        )
        heldout_output = np.asarray(source["output_map"], dtype=float)
    generators = {"primary": primary_generator, "heldout": heldout_generator}
    outputs = {"primary": primary_output, "heldout": heldout_output}
    blocks = {}
    block_metrics = {}
    for anchor in ("primary", "heldout"):
        blocks[anchor], block_metrics[anchor] = _complete_blocks(
            generators[anchor],
            basis[anchor]["restriction"],
            basis[anchor]["lifting"],
            outputs[anchor],
        )
    numerical_gates = frozen["contract"]["complete_closure"]["numerical_gates"]
    coordinate_passed = bool(
        all(
            metrics["coordinate_reconstruction_relative_defect"]
            <= numerical_gates["coordinate_reconstruction_relative_defect_max"]
            and metrics["restriction_lifting_identity_defect"] <= 5.0e-10
            and metrics["restriction_stable_annihilation_defect"] <= 5.0e-10
            and metrics["stable_lifting_annihilation_defect"] <= 5.0e-10
            and metrics["stable_basis_orthogonality_defect"] <= 5.0e-10
            and metrics["stable_spectral_abscissa_per_second"]
            <= -manifest.STABILITY_MARGIN_PER_SECOND
            for metrics in block_metrics.values()
        )
    )
    with np.load(
        generator_parent.manifest.R32_DIRECTORY / "R32_transfer.npz",
        allow_pickle=False,
    ) as source:
        frequencies = np.asarray(
            source["angular_frequencies_per_second"], dtype=float
        )
    heldout_frequencies = memory_tools._heldout_frequencies(frequencies)
    prepared = {
        anchor: _prepare_memory(blocks[anchor], frequencies, heldout_frequencies)
        for anchor in ("primary", "heldout")
    }
    transfer_gates = frozen["contract"]["complete_closure"][
        "pass_requires_at_both_anchors_on_training_and_heldout"
    ]
    spectral_gates = frozen["contract"]["complete_closure"]["spectral_fidelity"]
    spectral_gates = {
        **spectral_gates,
        "nonstable_threshold_per_second": -manifest.STABILITY_MARGIN_PER_SECOND,
    }
    candidate_metrics = []
    error_arrays = {}
    selected = None
    selected_arrays = None
    best = None
    best_arrays = None
    numerical_integrity = coordinate_passed
    for order in manifest.MEMORY_ORDERS:
        item = {
            "memory_order": order,
            "online_continuous_dimension": manifest.RESOLVED_DIMENSION + order,
        }
        model_arrays = {}
        anchor_passes = []
        for anchor in ("primary", "heldout"):
            metrics, arrays = _candidate(
                prepared[anchor],
                generators[anchor],
                frequencies,
                heldout_frequencies,
                order,
                transfer_gates,
                numerical_gates,
                spectral_gates,
            )
            item[anchor] = metrics
            anchor_passes.append(metrics["passed"])
            numerical_integrity &= metrics["numerical_passed"]
            for name, value in arrays.items():
                if name.endswith("_errors"):
                    error_arrays[f"M{order}_{anchor}_{name}"] = value
                else:
                    model_arrays[f"{anchor}_{name}"] = value
        item["joint_passed"] = bool(all(anchor_passes))
        item["maximum_gate_ratio"] = max(
            _candidate_score(
                item[anchor],
                transfer_gates,
                spectral_gates[
                    "bidirectional_nearest_eigenvalue_relative_defect_max"
                ],
            )
            for anchor in ("primary", "heldout")
        )
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
        numerical_integrity
        and np.isfinite(elapsed)
        and elapsed
        <= 3600.0
        * frozen["contract"]["execution_budget"]["maximum_wall_hours"]
    )
    classification, authorized_next, passed = _classification(
        numerical_passed, selected
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "metrics.json",
        {
            "coordinate_metrics": block_metrics,
            "coordinate_passed": coordinate_passed,
            "candidate_metrics": candidate_metrics,
            "selected": selected,
            "best": best,
            "numerical_passed": numerical_passed,
            "wall_seconds": elapsed,
        },
    )
    _write_npz(
        CANONICAL_DIRECTORY / "candidate_errors.npz",
        training_angular_frequencies_per_second=frequencies,
        heldout_angular_frequencies_per_second=heldout_frequencies,
        **error_arrays,
    )
    _write_npz(
        CANONICAL_DIRECTORY / "decisive_model.npz",
        **(
            selected_arrays
            if selected_arrays is not None
            else (best_arrays if best_arrays is not None else {})
        ),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "numerical_passed": numerical_passed,
        "coordinate_passed": coordinate_passed,
        "selected_memory_order": None if selected is None else selected["memory_order"],
        "selected_online_continuous_dimension": (
            None if selected is None else selected["online_continuous_dimension"]
        ),
        "selected_maximum_gate_ratio": (
            None if selected is None else selected["maximum_gate_ratio"]
        ),
        "best_memory_order": None if best is None else best["memory_order"],
        "best_maximum_gate_ratio": None if best is None else best["maximum_gate_ratio"],
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 0,
        "new_truth_anchors": 0,
        "physical_failure_detected": False,
        "production_coefficients_authorized": False,
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
            "parent_package_hashes": _checksums(parent.CANONICAL_DIRECTORY),
            "generator_package_hashes": _checksums(
                generator_parent.CANONICAL_DIRECTORY
            ),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
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
    if selected is not None:
        detail = (
            f"Selected complete-closure memory order `{selected['memory_order']}` "
            f"and online dimension `{selected['online_continuous_dimension']}`; "
            f"the maximum normalized gate ratio is "
            f"`{selected['maximum_gate_ratio']:.6e}`."
        )
    else:
        detail = (
            f"No complete closure passed through order 124. The best order was "
            f"`{best['memory_order'] if best else None}` with maximum normalized "
            f"gate ratio `{best['maximum_gate_ratio'] if best else None}`."
        )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Complete resolved-closure audit WP10c9d6c7c3b5c4f25s",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "This saved-generator audit tested the stable memory's feedback into all R196 resolved derivatives together with the conservative face-flux output. It executed no truth assembly, nonlinear root, or propagation.",
                "",
                detail,
                "",
                f"Authorized next artifact: `{authorized_next}`. Production coefficients, a predictive cycle, and reduced slow evolution remain blocked.",
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
