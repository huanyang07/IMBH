#!/usr/bin/env python3
"""Execute the saved-generator rank-adaptive common-memory audit."""

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

import run_causal_inner_common_resolved_subspace_cross_anchor_preflight_wp10c9d6c7c3b5c4f25o as parent  # noqa: E402
import run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1 as history  # noqa: E402
import run_causal_inner_invariant_projection_spectrum_audit_wp10c9d6c7c3b5c4f25e as projection_tools  # noqa: E402
import run_causal_inner_larger_coarse_pde_audit_wp10c9d6c7c3b5c4f25k as r32_tools  # noqa: E402
import run_causal_inner_rank_adaptive_common_memory_manifest_wp10c9d6c7c3b5c4f25p as manifest  # noqa: E402
import run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m as memory_tools  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    _descriptor_matrices,
    _node_reconstruction_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25q"
MANIFEST_COMMIT = "0c387058b937cd831743394c034c72c47a1f8efc"
MANIFEST_PARENT = "e4c0276eed0329185cf168a3a5fb46b28df6ced3"
MANIFEST_TREE = "96cfacaed8c259de1e3cab376530a53acfc57c53"

ARTIFACT = (
    "causal_inner_rank_adaptive_common_memory_audit_"
    "wp10c9d6c7c3b5c4f25q"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_rank_adaptive_common_memory_audit_"
    "wp10c9d6c7c3b5c4f25q.py"
)
THIS_TEST = (
    "tests/test_causal_inner_rank_adaptive_common_memory_audit_"
    "wp10c9d6c7c3b5c4f25q.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_RANK_ADAPTIVE_COMMON_MEMORY_"
    "AUDIT_WP10C9D6C7C3B5C4F25Q_2026-08-18.md"
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
    "two_anchor_rank_adaptive_common_memory_passed_"
    "online_prototype_manifest_authorized"
)
CAP_FAIL_CLASSIFICATION = (
    "common_memory_cap_failed_local_fiber_parametric_memory_"
    "architecture_manifest_authorized"
)
CHART_FAIL_CLASSIFICATION = (
    "common_resolved_chart_failed_local_fiber_atlas_manifest_authorized"
)
NUMERICAL_FAIL_CLASSIFICATION = (
    "rank_adaptive_common_memory_numerical_failure_stop"
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


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
        raise RuntimeError("rank-adaptive manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("rank-adaptive manifest parent changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("rank-adaptive manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["common_rank_ladder"]["candidates"]
        != list(manifest.COMMON_RANK_CANDIDATES)
        or contract["execution_budget"][
            "allowed_new_full_560_direction_generator_assemblies"
        ]
        != 0
    ):
        raise RuntimeError("rank-adaptive execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(parent.CANONICAL_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent decisive input changed: {name}")
    _checksums(parent.CANONICAL_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("rank-adaptive audit requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _physical_projection(data: dict, a2_dual: np.ndarray) -> dict[str, np.ndarray]:
    state = np.asarray(data["state"], dtype=float)
    columns = np.asarray(data["columns"], dtype=float)
    rows = np.asarray(data["rows"], dtype=float)
    (
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        _,
        _,
    ) = _node_reconstruction_weights(data["context"], state)
    mapped, height = _descriptor_matrices(
        data["context"],
        state,
        columns,
        rows,
        node_weights,
        node_cells,
        node_radii,
        node_measures,
    )
    storage, _, _ = r32_tools._R32_storage_restriction(mapped, height, rows)
    restriction, lifting, complement, metrics = (
        projection_tools._complete_qr_projection(storage, a2_dual)
    )
    if (
        metrics["resolved_rank"] != manifest.PHYSICAL_R32_DIMENSION
        or metrics["restriction_lifting_identity_defect"] > 5.0e-11
        or metrics["restriction_complement_annihilation_defect"] > 5.0e-11
        or metrics["complement_orthogonality_defect"] > 5.0e-11
    ):
        raise RuntimeError("saved-anchor physical projection changed")
    return {
        "restriction": restriction,
        "lifting": lifting,
        "complement": complement,
    }


def _rank_pass(
    primary_metrics: dict,
    heldout_metrics: dict,
    cross_cosine: float,
    gates: dict,
) -> bool:
    def anchor_pass(metrics: dict) -> bool:
        return bool(
            metrics["common_modal_basis_orthogonality_defect"]
            <= gates["common_basis_orthogonality_defect_max"]
            and metrics["common_augmented_restriction_lifting_identity_defect"]
            <= gates["augmented_restriction_lifting_identity_defect_max"]
            and metrics["common_augmented_restriction_stable_annihilation_defect"]
            <= gates["augmented_restriction_stable_annihilation_defect_max"]
            and metrics["remaining_common_unresolved_spectral_abscissa_per_second"]
            <= gates[
                "remaining_unresolved_spectral_abscissa_per_second_max"
            ]
        )

    return bool(
        anchor_pass(primary_metrics)
        and anchor_pass(heldout_metrics)
        and cross_cosine
        >= gates["minimum_cross_anchor_basis_principal_cosine"]
    )


def _prepare_memory(
    arrays: dict[str, np.ndarray],
    frequencies: np.ndarray,
    heldout_frequencies: np.ndarray,
) -> dict:
    normalized = memory_tools._normalize_system(
        arrays["stable_forcing"],
        arrays["stable_observation"],
        arrays["direct"],
    )
    forcing, observation, direct, input_scales, output_scales = normalized
    training_reference, training_reference_residual = (
        memory_tools._frequency_response(
            arrays["stable_operator"],
            forcing,
            observation,
            direct,
            frequencies,
        )
    )
    heldout_reference, heldout_reference_residual = (
        memory_tools._frequency_response(
            arrays["stable_operator"],
            forcing,
            observation,
            direct,
            heldout_frequencies,
        )
    )
    balanced, full_metrics = memory_tools._balanced_realization(
        arrays["stable_operator"], forcing, observation
    )
    return {
        "operator": arrays["stable_operator"],
        "forcing": forcing,
        "observation": observation,
        "direct": direct,
        "input_scales": input_scales,
        "output_scales": output_scales,
        "training_reference": training_reference,
        "heldout_reference": heldout_reference,
        "balanced": balanced,
        "full_metrics": full_metrics,
        "reference_frequency_residual": max(
            training_reference_residual, heldout_reference_residual
        ),
    }


def _memory_candidate(
    prepared: dict,
    frequencies: np.ndarray,
    heldout_frequencies: np.ndarray,
    order: int,
    gates: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    operator, forcing, observation, truncation = memory_tools._truncate_balanced(
        prepared["operator"],
        prepared["forcing"],
        prepared["observation"],
        prepared["balanced"],
        int(order),
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
    training_dynamic, training_total, training_errors = memory_tools._error_metrics(
        training_approximation,
        prepared["training_reference"],
        prepared["direct"],
    )
    heldout_dynamic, heldout_total, heldout_errors = memory_tools._error_metrics(
        heldout_approximation,
        prepared["heldout_reference"],
        prepared["direct"],
    )
    metrics = {
        "memory_order": int(order),
        **prepared["full_metrics"],
        **truncation,
        **memory_tools._stability_metrics(operator),
        **{f"training_{key}": value for key, value in training_errors.items()},
        **{f"heldout_{key}": value for key, value in heldout_errors.items()},
        "maximum_frequency_solve_relative_residual": max(
            prepared["reference_frequency_residual"],
            training_residual,
            heldout_residual,
        ),
    }
    metrics["training_passed"] = memory_tools._gates_pass(
        metrics, gates, "training"
    )
    metrics["heldout_passed"] = memory_tools._gates_pass(
        metrics, gates, "heldout"
    )
    metrics["passed"] = bool(
        metrics["training_passed"]
        and metrics["heldout_passed"]
        and metrics["controllability_gramian_relative_residual"] <= 1.0e-8
        and metrics["observability_gramian_relative_residual"] <= 1.0e-8
        and metrics["maximum_frequency_solve_relative_residual"] <= 1.0e-10
    )
    arrays = {
        "reduced_operator": operator,
        "reduced_forcing": forcing,
        "reduced_observation": observation,
        "normalized_direct": prepared["direct"],
        "input_scales": prepared["input_scales"],
        "output_scales": prepared["output_scales"],
        "training_dynamic_errors": training_dynamic,
        "training_total_errors": training_total,
        "heldout_dynamic_errors": heldout_dynamic,
        "heldout_total_errors": heldout_total,
    }
    return metrics, arrays


def _candidate_score(metrics: dict, gates: dict) -> float:
    ratios = []
    for prefix in ("training", "heldout"):
        for name in (
            "maximum_normalized_dynamic_transfer_relative_error",
            "RMS_normalized_dynamic_transfer_relative_error",
            "DC_normalized_dynamic_transfer_relative_error",
            "maximum_normalized_total_transfer_relative_error",
            "RMS_normalized_total_transfer_relative_error",
            "DC_normalized_total_transfer_relative_error",
        ):
            ratios.append(metrics[f"{prefix}_{name}"] / gates[f"{name}_max"])
    return float(max(ratios))


def _classification(
    numerical_passed: bool,
    stable_rank_count: int,
    selected: dict | None,
) -> tuple[str, str | None, bool]:
    if not numerical_passed:
        return NUMERICAL_FAIL_CLASSIFICATION, None, False
    if selected is not None:
        return (
            PASS_CLASSIFICATION,
            "definitions_only_R32_rank_adaptive_memory_online_prototype_manifest",
            True,
        )
    if stable_rank_count:
        return (
            CAP_FAIL_CLASSIFICATION,
            "definitions_only_local_fiber_parametric_memory_architecture_manifest",
            False,
        )
    return (
        CHART_FAIL_CLASSIFICATION,
        "definitions_only_local_fiber_atlas_manifest",
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
        raise RuntimeError("rank-adaptive audit is already canonicalized")
    began = time.perf_counter()
    primary_data = history._state_data("primary_20ms")
    heldout_data = history._state_data("heldout_16ms")
    with np.load(
        parent.manifest.GENERATOR_DIRECTORY / "descriptor_A.npz",
        allow_pickle=False,
    ) as source:
        primary_generator = np.asarray(
            source["complete_fixed_Q_generator"], dtype=float
        )
    with np.load(
        parent.manifest.GENERATOR_DIRECTORY / "projection.npz",
        allow_pickle=False,
    ) as source:
        primary_a2 = np.asarray(source["a2_dual"], dtype=float)
        primary_output = np.asarray(source["output_map"], dtype=float)
    with np.load(
        parent.CANONICAL_DIRECTORY / "heldout_generator.npz",
        allow_pickle=False,
    ) as source:
        heldout_generator = np.asarray(
            source["complete_fixed_Q_generator"], dtype=float
        )
        heldout_a2 = np.asarray(source["a2_dual"], dtype=float)
        heldout_output = np.asarray(source["output_map"], dtype=float)
        if not np.array_equal(source["primitive_state"], heldout_data["state"]):
            raise RuntimeError("saved held-out state changed")
    primary_projection = _physical_projection(primary_data, primary_a2)
    heldout_projection = _physical_projection(heldout_data, heldout_a2)
    with np.load(
        parent.CANONICAL_DIRECTORY / "common_subspace.npz",
        allow_pickle=False,
    ) as source:
        reference = np.asarray(source["common_reference_basis"], dtype=float)
        primary_local = np.asarray(
            source["primary_local_promoted_basis"], dtype=float
        )
        heldout_local = np.asarray(
            source["heldout_local_promoted_basis"], dtype=float
        )
    with np.load(
        parent.manifest.R32_DIRECTORY / "R32_transfer.npz", allow_pickle=False
    ) as source:
        frequencies = np.asarray(
            source["angular_frequencies_per_second"], dtype=float
        )
    heldout_frequencies = memory_tools._heldout_frequencies(frequencies)
    rank_gates = frozen["contract"]["common_rank_ladder"][
        "pass_requires_at_both_anchors"
    ]
    memory_gates = frozen["contract"]["memory_order_ladder"][
        "pass_requires_at_both_anchors"
    ]
    rank_metrics = []
    candidate_metrics = []
    error_arrays = {}
    stable_rank_count = 0
    selected = None
    selected_arrays = None
    selected_basis_arrays = None
    best = None
    best_arrays = None
    best_basis_arrays = None
    numerical_integrity = True

    for rank in manifest.COMMON_RANK_CANDIDATES:
        basis = reference[:, :rank]
        primary_common, primary_common_metrics = parent._anchor_common_basis(
            basis,
            primary_projection["complement"],
            primary_projection["lifting"],
            primary_projection["restriction"],
            primary_local,
            primary_generator,
            primary_output,
        )
        heldout_common, heldout_common_metrics = parent._anchor_common_basis(
            basis,
            heldout_projection["complement"],
            heldout_projection["lifting"],
            heldout_projection["restriction"],
            heldout_local,
            heldout_generator,
            heldout_output,
        )
        cross_cosines = np.linalg.svd(
            primary_common["aligned_common_basis"].T
            @ heldout_common["aligned_common_basis"],
            compute_uv=False,
        )
        rank_passed = _rank_pass(
            primary_common_metrics,
            heldout_common_metrics,
            float(np.min(cross_cosines)),
            rank_gates,
        )
        current_rank_metrics = {
            "common_rank": rank,
            "rank_passed": rank_passed,
            "cross_anchor_minimum_principal_cosine": float(
                np.min(cross_cosines)
            ),
            "cross_anchor_maximum_principal_angle_degrees": float(
                np.degrees(
                    np.arccos(np.clip(np.min(cross_cosines), -1.0, 1.0))
                )
            ),
            "primary": primary_common_metrics,
            "heldout": heldout_common_metrics,
        }
        rank_metrics.append(current_rank_metrics)
        if not rank_passed:
            continue
        stable_rank_count += 1
        primary_prepared = _prepare_memory(
            primary_common, frequencies, heldout_frequencies
        )
        heldout_prepared = _prepare_memory(
            heldout_common, frequencies, heldout_frequencies
        )
        for prepared in (primary_prepared, heldout_prepared):
            numerical_integrity &= bool(
                prepared["full_metrics"][
                    "controllability_gramian_relative_residual"
                ]
                <= 1.0e-8
                and prepared["full_metrics"][
                    "observability_gramian_relative_residual"
                ]
                <= 1.0e-8
                and prepared["reference_frequency_residual"] <= 1.0e-10
            )
        for order in manifest._memory_orders(rank):
            primary_candidate, primary_arrays = _memory_candidate(
                primary_prepared,
                frequencies,
                heldout_frequencies,
                order,
                memory_gates,
            )
            heldout_candidate, heldout_arrays = _memory_candidate(
                heldout_prepared,
                frequencies,
                heldout_frequencies,
                order,
                memory_gates,
            )
            online_dimension = (
                manifest.PHYSICAL_R32_DIMENSION + rank + order
            )
            joint_passed = bool(
                primary_candidate["passed"]
                and heldout_candidate["passed"]
                and online_dimension
                <= manifest.MAXIMUM_ONLINE_CONTINUOUS_DIMENSION
            )
            score = max(
                _candidate_score(primary_candidate, memory_gates),
                _candidate_score(heldout_candidate, memory_gates),
            )
            item = {
                "common_rank": rank,
                "memory_order": order,
                "online_continuous_dimension": online_dimension,
                "joint_passed": joint_passed,
                "maximum_gate_ratio": score,
                "primary": primary_candidate,
                "heldout": heldout_candidate,
            }
            candidate_metrics.append(item)
            prefix = f"R{rank}_M{order}"
            for anchor, values in (
                ("primary", primary_arrays),
                ("heldout", heldout_arrays),
            ):
                for name in (
                    "training_dynamic_errors",
                    "training_total_errors",
                    "heldout_dynamic_errors",
                    "heldout_total_errors",
                ):
                    error_arrays[f"{prefix}_{anchor}_{name}"] = values[name]
            model_arrays = {}
            for anchor, values in (
                ("primary", primary_arrays),
                ("heldout", heldout_arrays),
            ):
                for name in (
                    "reduced_operator",
                    "reduced_forcing",
                    "reduced_observation",
                    "normalized_direct",
                    "input_scales",
                    "output_scales",
                ):
                    model_arrays[f"{anchor}_{name}"] = values[name]
            basis_arrays = {
                "reference_basis": basis,
                "primary_aligned_common_basis": primary_common[
                    "aligned_common_basis"
                ],
                "heldout_aligned_common_basis": heldout_common[
                    "aligned_common_basis"
                ],
                "primary_augmented_restriction": primary_common[
                    "augmented_restriction"
                ],
                "primary_augmented_lifting": primary_common[
                    "augmented_lifting"
                ],
                "heldout_augmented_restriction": heldout_common[
                    "augmented_restriction"
                ],
                "heldout_augmented_lifting": heldout_common[
                    "augmented_lifting"
                ],
            }
            if best is None or score < best["maximum_gate_ratio"]:
                best = item
                best_arrays = model_arrays
                best_basis_arrays = basis_arrays
            if joint_passed:
                selected = item
                selected_arrays = model_arrays
                selected_basis_arrays = basis_arrays
                break
        if selected is not None:
            break

    elapsed = float(time.perf_counter() - began)
    maximum_seconds = 3600.0 * frozen["contract"]["execution_budget"][
        "maximum_wall_hours"
    ]
    numerical_passed = bool(
        np.isfinite(elapsed)
        and elapsed <= maximum_seconds
        and numerical_integrity
        and len(rank_metrics) >= 1
        and all(np.isfinite(item["cross_anchor_minimum_principal_cosine"]) for item in rank_metrics)
    )
    classification, authorized_next, passed = _classification(
        numerical_passed, stable_rank_count, selected
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "metrics.json",
        {
            "rank_metrics": rank_metrics,
            "candidate_metrics": candidate_metrics,
            "stable_rank_count": stable_rank_count,
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
    decisive_arrays = selected_arrays if selected_arrays is not None else best_arrays
    decisive_basis = (
        selected_basis_arrays
        if selected_basis_arrays is not None
        else best_basis_arrays
    )
    _write_npz(
        CANONICAL_DIRECTORY / "decisive_model.npz",
        **({} if decisive_arrays is None else decisive_arrays),
    )
    _write_npz(
        CANONICAL_DIRECTORY / "decisive_basis.npz",
        **({} if decisive_basis is None else decisive_basis),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "numerical_passed": numerical_passed,
        "stable_rank_count": stable_rank_count,
        "selected_common_rank": None if selected is None else selected["common_rank"],
        "selected_memory_order": None if selected is None else selected["memory_order"],
        "selected_online_continuous_dimension": (
            None if selected is None else selected["online_continuous_dimension"]
        ),
        "best_common_rank": None if best is None else best["common_rank"],
        "best_memory_order": None if best is None else best["memory_order"],
        "best_maximum_gate_ratio": None if best is None else best["maximum_gate_ratio"],
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 0,
        "new_truth_anchors": 0,
        "physical_failure_detected": False,
        "production_coefficients_authorized": False,
        "online_reduced_solver_implementation_authorized": False,
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
            f"Selected common rank `{selected['common_rank']}`, memory order "
            f"`{selected['memory_order']}`, and online dimension "
            f"`{selected['online_continuous_dimension']}`. The maximum normalized "
            f"gate ratio is `{selected['maximum_gate_ratio']:.6e}`."
        )
    elif best is not None:
        detail = (
            f"No joint candidate passed within R320. The best candidate was common "
            f"rank `{best['common_rank']}` with memory order `{best['memory_order']}` "
            f"and maximum normalized gate ratio `{best['maximum_gate_ratio']:.6e}`."
        )
    else:
        detail = "No common rank stabilized both saved anchor generators."
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Rank-adaptive common-memory audit WP10c9d6c7c3b5c4f25q",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "This audit reused the two hash-locked 560-dimensional generators. It executed no truth assembly, nonlinear root, or state propagation.",
                "",
                detail,
                "",
                f"Authorized next artifact: `{authorized_next}`. Production coefficients, an online predictive cycle, and reduced slow evolution remain blocked.",
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
