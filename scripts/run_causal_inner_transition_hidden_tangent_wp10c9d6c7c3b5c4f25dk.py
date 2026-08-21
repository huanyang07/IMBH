#!/usr/bin/env python3
"""Execute the saved-generator transition hidden-tangent diagnostic."""

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

import run_causal_inner_exact_geometric_470_chart_preflight_wp10c9d6c7c3b5c4f25de as chart  # noqa: E402
import run_causal_inner_transition_hidden_tangent_manifest_wp10c9d6c7c3b5c4f25dj as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dk"
PARENT_COMMIT = "cf20a5019d226ed2468dd0e1853acab415f6c980"
PARENT_PARENT = "32eee7a20e0be8c8ce9d12b036477a5ae389b909"
PARENT_TREE = "ca2bc1a86857d4ea8660a1c74fd96bc1e0519972"

RANK8_CLASSIFICATION = "common_rank8_transition_hidden_tangent_candidate_supported"
ENRICHED_CLASSIFICATION = (
    "rank_adaptive_common_transition_hidden_tangent_candidate_supported"
)
REDUCTION_REJECTED_CLASSIFICATION = (
    "transition_hidden_tangent_reduction_rejected_"
    "full470_offline_impulse_reference_required"
)
INFRASTRUCTURE_CLASSIFICATION = "transition_tangent_diagnostic_infrastructure_failed"
AUTHORIZED_NEXT_PASS = "WP10c9d6c7c3b5c4f25dl"
AUTHORIZED_NEXT_REJECTED = "WP10c9d6c7c3b5c4f25dl"

ARTIFACT = "causal_inner_transition_hidden_tangent_wp10c9d6c7c3b5c4f25dk"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_transition_hidden_tangent_"
    "wp10c9d6c7c3b5c4f25dk.py"
)
THIS_TEST = (
    "tests/test_causal_inner_transition_hidden_tangent_"
    "wp10c9d6c7c3b5c4f25dk.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRANSITION_HIDDEN_TANGENT_"
    "WP10C9D6C7C3B5C4F25DK_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_DIRECTORY = manifest.CANONICAL_DIRECTORY
PARENT_CONTRACT = PARENT_DIRECTORY / "transition_hidden_tangent_contract.json"
PARENT_BASIS_ARRAYS = manifest.parent.CANONICAL_DIRECTORY / "hidden_basis_screen_arrays.npz"
CURRENT_CHART_ARRAYS = manifest.parent.EXACT_CHART_ARRAYS
FIELD_DIRECTORY = chart.manifest.parent.field_manifest.CANONICAL_DIRECTORY
FIELD_ARRAYS = chart.manifest.parent.FIELD_ARRAYS


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


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("transition-tangent execution parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("transition-tangent execution lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("transition-tangent execution tree changed")
    parent_hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    contract = _read(PARENT_CONTRACT)
    provenance = _read(PARENT_DIRECTORY / "provenance.json")
    basis_provenance = _read(
        manifest.parent.CANONICAL_DIRECTORY / "provenance.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["saved_complete_generator_reuse_frozen"]
        or summary["old_rejected_resolved_lifting_reused"]
        or summary["new_exact_fixed_Q_rate_calls"] != 0
        or summary["new_complete_generator_assemblies"] != 0
        or summary["new_nonlinear_roots"] != 0
        or summary["propagated_states"] != 0
        or summary["sealed_16ms_opened"]
        or contract["execution_budget"][
            "new_exact_fixed_Q_rate_evaluations_equal"
        ]
        != 0
        or contract["execution_budget"][
            "new_complete_generator_assemblies_equal"
        ]
        != 0
        or contract["rank_adaptive_fallback"]["maximum_rank"]
        != manifest.MAXIMUM_HIDDEN_RANK
    ):
        raise RuntimeError("transition hidden-tangent contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"transition-tangent source changed: {relative}")
    for name, path in {
        "parent_summary": manifest.PARENT_SUMMARY,
        "parent_contract": manifest.PARENT_CONTRACT,
        "parent_arrays": manifest.PARENT_ARRAYS,
        "saved_generator": manifest.SAVED_GENERATOR,
        "saved_generator_metrics": manifest.SAVED_GENERATOR_METRICS,
        "saved_generator_provenance": manifest.SAVED_GENERATOR_PROVENANCE,
        "saved_generator_seed_lock": manifest.SAVED_GENERATOR_SEED_LOCK,
        "saved_checkpoint": manifest.SAVED_CHECKPOINT,
        "exact_chart_runner": ROOT / manifest.EXACT_CHART_RUNNER,
        "fixed_Q_source": ROOT / manifest.FIXED_Q_SOURCE,
        "tangent_source": ROOT / manifest.TANGENT_SOURCE,
    }.items():
        if _sha(path) != contract["decisive_input_hashes"][name]:
            raise RuntimeError(f"transition-tangent decisive input changed: {name}")
    field_hashes = _checksums(FIELD_DIRECTORY)
    if field_hashes[FIELD_ARRAYS.name] != _sha(FIELD_ARRAYS):
        raise RuntimeError("exact chart field arrays changed")
    if (
        _sha(CURRENT_CHART_ARRAYS)
        != basis_provenance["decisive_input_hashes"]["exact_chart_arrays"]
    ):
        raise RuntimeError("current exact chart arrays changed")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("transition-tangent execution requires a clean tracked tree")
    return {
        "parent_hashes": parent_hashes,
        "field_hashes": field_hashes,
        "parent_classification": summary["classification"],
    }


def _canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _model():
    field_module = chart.manifest.parent.field_manifest
    field = field_module.ForwardQuadraticAuthenticCenterField(
        _load_npz(FIELD_ARRAYS)
    )
    return field.model


def _energy_capture(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    samples = np.atleast_2d(np.asarray(vectors, dtype=float))
    denominator = np.maximum(np.sum(samples**2, axis=1), np.finfo(float).tiny)
    return np.sum((samples @ basis) ** 2, axis=1) / denominator


def _physical_capture(
    augmented: np.ndarray,
    hidden_response: np.ndarray,
    projected_hidden_response: np.ndarray,
    hidden_lifting: np.ndarray,
) -> tuple[float, float]:
    zero = np.zeros((manifest.GAUGE_DIMENSION, hidden_response.shape[1]))
    physical = np.linalg.solve(
        augmented, np.vstack((hidden_lifting @ hidden_response, zero))
    )
    projected = np.linalg.solve(
        augmented,
        np.vstack((hidden_lifting @ projected_hidden_response, zero)),
    )
    denominator = max(float(np.linalg.norm(physical)), np.finfo(float).tiny)
    relative_error = float(np.linalg.norm(projected - physical) / denominator)
    return float(1.0 - relative_error**2), relative_error


def _candidate_metrics(
    *,
    basis: np.ndarray,
    hidden_response: np.ndarray,
    augmented: np.ndarray,
    hidden_lifting: np.ndarray,
    macro_restriction: np.ndarray,
    seed_hidden_rates: np.ndarray,
    current_hidden_rate: np.ndarray,
    current_augmented_chart: np.ndarray,
) -> dict:
    projected_response = basis @ (basis.T @ hidden_response)
    response_scale = max(
        float(np.linalg.norm(hidden_response)), np.finfo(float).tiny
    )
    physical_capture, physical_error = _physical_capture(
        augmented,
        hidden_response,
        projected_response,
        hidden_lifting,
    )
    reduced = basis.T @ hidden_response
    eigenvalues = np.linalg.eigvals(reduced)
    projected_current_hidden = basis @ (basis.T @ current_hidden_rate)
    current_zero = np.zeros(manifest.GAUGE_DIMENSION)
    current_physical = np.linalg.solve(
        current_augmented_chart,
        np.concatenate((hidden_lifting @ current_hidden_rate, current_zero)),
    )
    projected_current_physical = np.linalg.solve(
        current_augmented_chart,
        np.concatenate(
            (hidden_lifting @ projected_current_hidden, current_zero)
        ),
    )
    current_physical_scale = max(
        float(np.linalg.norm(current_physical)), np.finfo(float).tiny
    )
    current_physical_error = float(
        np.linalg.norm(projected_current_physical - current_physical)
        / current_physical_scale
    )
    return {
        "rank": int(basis.shape[1]),
        "hidden_tangent_invariance_relative_defect": float(
            np.linalg.norm(hidden_response - projected_response) / response_scale
        ),
        "hidden_physical_tangent_energy_capture": physical_capture,
        "hidden_physical_tangent_relative_error": physical_error,
        "minimum_seed_rate_action_energy_capture": float(
            np.min(_energy_capture(seed_hidden_rates, basis))
        ),
        "current_primary_rate_action_energy_capture": float(
            _energy_capture(current_hidden_rate, basis)[0]
        ),
        "current_primary_gauge_fixed_physical_action_energy_capture": float(
            1.0 - current_physical_error**2
        ),
        "current_primary_gauge_fixed_physical_action_relative_error": (
            current_physical_error
        ),
        "basis_orthonormality_infinity_defect": float(
            np.linalg.norm(
                basis.T @ basis - np.eye(basis.shape[1]), ord=np.inf
            )
        ),
        "action_macro_annihilation_infinity_defect": float(
            np.linalg.norm(macro_restriction @ hidden_lifting @ basis, ord=np.inf)
        ),
        "reduced_tangent_condition_number": float(np.linalg.cond(reduced)),
        "reduced_tangent_eigenvalues_real_per_second": np.real(eigenvalues),
        "reduced_tangent_eigenvalues_imaginary_per_second": np.imag(eigenvalues),
        "reduced_nonstable_eigenvalue_count_diagnostic": int(
            np.count_nonzero(np.real(eigenvalues) >= 0.0)
        ),
    }


def _candidate_passes(metrics: dict) -> bool:
    return bool(
        metrics["hidden_tangent_invariance_relative_defect"]
        <= manifest.TANGENT_INVARIANCE_GATE
        and metrics["hidden_physical_tangent_energy_capture"]
        >= manifest.PHYSICAL_TANGENT_CAPTURE_GATE
        and metrics["minimum_seed_rate_action_energy_capture"]
        >= manifest.parent.TRAINING_CAPTURE_GATE
        and metrics["current_primary_rate_action_energy_capture"]
        >= manifest.parent.CURRENT_HIDDEN_CAPTURE_GATE
        and metrics[
            "current_primary_gauge_fixed_physical_action_energy_capture"
        ]
        >= manifest.parent.CURRENT_PHYSICAL_CAPTURE_GATE
        and metrics["basis_orthonormality_infinity_defect"]
        <= manifest.ORTHONORMALITY_GATE
        and metrics["action_macro_annihilation_infinity_defect"]
        <= manifest.MACRO_ANNIHILATION_GATE
        and metrics["rank"] <= manifest.MAXIMUM_HIDDEN_RANK
    )


class _CoordinateOperator:
    def __init__(
        self,
        *,
        model,
        state: np.ndarray,
        coordinate_jacobian: np.ndarray,
        gauge_basis: np.ndarray,
        generator: np.ndarray,
        base_rate: np.ndarray,
        hidden_lifting: np.ndarray,
        hidden_dual: np.ndarray,
    ) -> None:
        self.model = model
        self.state = np.asarray(state, dtype=float)
        self.coordinate_jacobian = np.asarray(coordinate_jacobian, dtype=float)
        self.gauge_basis = np.asarray(gauge_basis, dtype=float)
        self.augmented = np.vstack(
            (self.coordinate_jacobian, self.gauge_basis.T)
        )
        self.generator = np.asarray(generator, dtype=float)
        self.base_rate = np.asarray(base_rate, dtype=float)
        self.hidden_lifting = np.asarray(hidden_lifting, dtype=float)
        self.hidden_dual = np.asarray(hidden_dual, dtype=float)
        self.anchor_delta = chart._delta(model, self.state)
        self.coordinate_jacobian_evaluations = 1

    def apply(
        self, hidden_basis: np.ndarray, step: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        basis = np.asarray(hidden_basis, dtype=float)
        coordinate_directions = self.hidden_lifting @ basis
        zero = np.zeros((manifest.GAUGE_DIMENSION, basis.shape[1]))
        physical_directions = np.linalg.solve(
            self.augmented, np.vstack((coordinate_directions, zero))
        )
        hessian_actions = []
        scales = []
        for column in range(basis.shape[1]):
            physical = physical_directions[:, column]
            scale = manifest.SIGNED_PHYSICAL_COMPONENT_RADIUS / max(
                float(np.max(np.abs(physical))), np.finfo(float).tiny
            )
            scaled = scale * physical
            plus = chart._state_from_delta(
                self.model, self.anchor_delta + step * scaled
            )
            minus = chart._state_from_delta(
                self.model, self.anchor_delta - step * scaled
            )
            plus_jacobian, _ = chart._coordinate_jacobian(self.model, plus)
            minus_jacobian, _ = chart._coordinate_jacobian(self.model, minus)
            self.coordinate_jacobian_evaluations += 2
            if (
                self.coordinate_jacobian_evaluations
                > manifest.MAXIMUM_COORDINATE_JACOBIAN_EVALUATIONS
            ):
                raise RuntimeError("coordinate-Jacobian budget exceeded")
            hessian_actions.append(
                ((plus_jacobian - minus_jacobian) / (2.0 * step) @ self.base_rate)
                / scale
            )
            scales.append(scale)
        hessian = np.column_stack(hessian_actions)
        coordinate_response = (
            self.coordinate_jacobian @ (self.generator @ physical_directions)
            + hessian
        )
        hidden_response = self.hidden_dual @ coordinate_response
        return hidden_response, coordinate_response, hessian, np.asarray(scales)


def _new_residual_directions(
    basis: np.ndarray, hidden_response: np.ndarray, count: int
) -> np.ndarray:
    residual = hidden_response - basis @ (basis.T @ hidden_response)
    left, singular, _ = np.linalg.svd(residual, full_matrices=False)
    tolerance = max(float(singular[0]), np.finfo(float).tiny) * 1.0e-12
    available = int(np.count_nonzero(singular > tolerance))
    take = min(int(count), available)
    if take == 0:
        return np.empty((basis.shape[0], 0), dtype=float)
    proposed = left[:, :take]
    proposed -= basis @ (basis.T @ proposed)
    orthogonal, _ = np.linalg.qr(proposed, mode="reduced")
    return _canonicalize_columns(orthogonal[:, :take])


def _execute() -> tuple[dict, dict[str, np.ndarray], dict]:
    began = time.perf_counter()
    basis_data = _load_npz(PARENT_BASIS_ARRAYS)
    current_chart = _load_npz(CURRENT_CHART_ARRAYS)
    generator_data = _load_npz(manifest.SAVED_GENERATOR)
    checkpoint = _load_npz(manifest.SAVED_CHECKPOINT)
    model = _model()

    state = np.asarray(checkpoint["current_primitive_charts"], dtype=float)
    checkpoint_columns = np.asarray(
        checkpoint["solver_primitive_column_scales"], dtype=float
    )
    columns_bitwise = bool(np.array_equal(checkpoint_columns, model.columns))
    coordinate_jacobian, chart_metrics = chart._coordinate_jacobian(model, state)
    gauge_basis = chart._canonical_null_basis(coordinate_jacobian)
    augmented = np.vstack((coordinate_jacobian, gauge_basis.T))
    augmented_condition = float(np.linalg.cond(augmented))

    B0 = np.asarray(basis_data["selected_hidden_basis388"], dtype=float)
    Z = np.asarray(basis_data["hidden_basis_Z388"], dtype=float)
    Q = np.asarray(basis_data["hidden_dual_Q388"], dtype=float)
    R = np.asarray(basis_data["macro_restriction_R82"], dtype=float)
    seed_hidden = np.asarray(basis_data["seed_hidden_rates388_per_s"], dtype=float)
    current_hidden = np.asarray(
        basis_data["current_primary_hidden_rate388_per_s"], dtype=float
    )
    current_augmented = np.asarray(
        current_chart["anchor_augmented_chart_jacobian"], dtype=float
    )
    generator = np.asarray(
        generator_data["complete_fixed_Q_generator"], dtype=float
    )
    base_rate = np.asarray(generator_data["fixed_Q_rate"], dtype=float)

    operator = _CoordinateOperator(
        model=model,
        state=state,
        coordinate_jacobian=coordinate_jacobian,
        gauge_basis=gauge_basis,
        generator=generator,
        base_rate=base_rate,
        hidden_lifting=Z,
        hidden_dual=Q,
    )

    step_hidden_responses = []
    step_coordinate_responses = []
    step_hessian_actions = []
    direction_scales = []
    for step in manifest.HESSIAN_STEPS:
        hidden, coordinate, hessian, scales = operator.apply(B0, step)
        step_hidden_responses.append(hidden)
        step_coordinate_responses.append(coordinate)
        step_hessian_actions.append(hessian)
        direction_scales.append(scales)
    adjacent_step_defects = []
    for coarse, fine in zip(
        step_hidden_responses[:-1], step_hidden_responses[1:], strict=True
    ):
        adjacent_step_defects.append(
            float(
                np.linalg.norm(coarse - fine)
                / max(np.linalg.norm(fine), np.finfo(float).tiny)
            )
        )
    maximum_step_defect = max(adjacent_step_defects)

    candidate_records = []
    selected_basis = B0
    selected_hidden_response = step_hidden_responses[-1]
    initial_metrics = _candidate_metrics(
        basis=selected_basis,
        hidden_response=selected_hidden_response,
        augmented=augmented,
        hidden_lifting=Z,
        macro_restriction=R,
        seed_hidden_rates=seed_hidden,
        current_hidden_rate=current_hidden,
        current_augmented_chart=current_augmented,
    )
    candidate_records.append(initial_metrics)

    infrastructure_passed = bool(
        columns_bitwise
        and chart_metrics["rank"] == manifest.COORDINATE_DIMENSION
        and augmented_condition <= manifest.CHART_CONDITION_GATE
        and maximum_step_defect <= manifest.TANGENT_STEP_CONVERGENCE_GATE
    )
    selected_passed = infrastructure_passed and _candidate_passes(initial_metrics)
    if infrastructure_passed and not selected_passed:
        for target_rank in manifest.ENRICHMENT_RANKS[1:]:
            while selected_basis.shape[1] < target_rank:
                need = target_rank - selected_basis.shape[1]
                new_directions = _new_residual_directions(
                    selected_basis, selected_hidden_response, need
                )
                if new_directions.shape[1] == 0:
                    break
                new_hidden, _, _, _ = operator.apply(
                    new_directions, manifest.HESSIAN_STEPS[1]
                )
                selected_basis = np.column_stack(
                    (selected_basis, new_directions)
                )
                selected_hidden_response = np.column_stack(
                    (selected_hidden_response, new_hidden)
                )
            record = _candidate_metrics(
                basis=selected_basis,
                hidden_response=selected_hidden_response,
                augmented=augmented,
                hidden_lifting=Z,
                macro_restriction=R,
                seed_hidden_rates=seed_hidden,
                current_hidden_rate=current_hidden,
                current_augmented_chart=current_augmented,
            )
            candidate_records.append(record)
            if _candidate_passes(record):
                selected_passed = True
                break
            if selected_basis.shape[1] >= manifest.MAXIMUM_HIDDEN_RANK:
                break

    selected_metrics = candidate_records[-1]
    if not infrastructure_passed:
        classification = INFRASTRUCTURE_CLASSIFICATION
        passed = False
        authorized_next = None
    elif selected_passed and selected_metrics["rank"] == manifest.INITIAL_HIDDEN_RANK:
        classification = RANK8_CLASSIFICATION
        passed = True
        authorized_next = AUTHORIZED_NEXT_PASS
    elif selected_passed:
        classification = ENRICHED_CLASSIFICATION
        passed = True
        authorized_next = AUTHORIZED_NEXT_PASS
    else:
        classification = REDUCTION_REJECTED_CLASSIFICATION
        passed = False
        authorized_next = AUTHORIZED_NEXT_REJECTED

    reduced_tangent = selected_basis.T @ selected_hidden_response
    metrics = {
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "checkpoint_time_seconds": float(checkpoint["elapsed_time_seconds"]),
        "checkpoint_completed_steps": int(checkpoint["completed_steps"]),
        "checkpoint_columns_bitwise_match_chart_model": columns_bitwise,
        "checkpoint_coordinate_chart": chart_metrics,
        "checkpoint_augmented_chart_condition_number": augmented_condition,
        "rank8_tangent_step_ladder": list(manifest.HESSIAN_STEPS),
        "rank8_adjacent_hidden_tangent_step_relative_defects": (
            adjacent_step_defects
        ),
        "maximum_rank8_hidden_tangent_step_relative_defect": maximum_step_defect,
        "candidate_rank_metrics": candidate_records,
        "selected_rank_metrics": selected_metrics,
        "coordinate_Jacobian_evaluations": (
            operator.coordinate_jacobian_evaluations
        ),
        "wall_seconds": float(time.perf_counter() - began),
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "new_chart_retractions": 0,
        "propagated_states": 0,
        "sealed_16ms_truth_calls": 0,
    }
    checks = {
        "checkpoint_columns": columns_bitwise,
        "coordinate_chart_rank": chart_metrics["rank"]
        == manifest.COORDINATE_DIMENSION,
        "coordinate_chart_condition": augmented_condition
        <= manifest.CHART_CONDITION_GATE,
        "rank8_tangent_step_convergence": maximum_step_defect
        <= manifest.TANGENT_STEP_CONVERGENCE_GATE,
        "selected_tangent_invariance": selected_metrics[
            "hidden_tangent_invariance_relative_defect"
        ]
        <= manifest.TANGENT_INVARIANCE_GATE,
        "selected_physical_tangent_capture": selected_metrics[
            "hidden_physical_tangent_energy_capture"
        ]
        >= manifest.PHYSICAL_TANGENT_CAPTURE_GATE,
        "selected_seed_rate_capture": selected_metrics[
            "minimum_seed_rate_action_energy_capture"
        ]
        >= manifest.parent.TRAINING_CAPTURE_GATE,
        "selected_current_rate_capture": selected_metrics[
            "current_primary_rate_action_energy_capture"
        ]
        >= manifest.parent.CURRENT_HIDDEN_CAPTURE_GATE,
        "selected_current_physical_capture": selected_metrics[
            "current_primary_gauge_fixed_physical_action_energy_capture"
        ]
        >= manifest.parent.CURRENT_PHYSICAL_CAPTURE_GATE,
        "selected_basis_orthonormality": selected_metrics[
            "basis_orthonormality_infinity_defect"
        ]
        <= manifest.ORTHONORMALITY_GATE,
        "selected_macro_annihilation": selected_metrics[
            "action_macro_annihilation_infinity_defect"
        ]
        <= manifest.MACRO_ANNIHILATION_GATE,
        "selected_rank_budget": selected_metrics["rank"]
        <= manifest.MAXIMUM_HIDDEN_RANK,
        "coordinate_Jacobian_budget": operator.coordinate_jacobian_evaluations
        <= manifest.MAXIMUM_COORDINATE_JACOBIAN_EVALUATIONS,
        "truth_budget": metrics["new_exact_fixed_Q_rate_calls"] == 0,
        "generator_budget": metrics["new_complete_generator_assemblies"] == 0,
        "root_budget": metrics["new_nonlinear_roots"] == 0,
        "retraction_budget": metrics["new_chart_retractions"] == 0,
        "propagation_budget": metrics["propagated_states"] == 0,
        "sealed_budget": metrics["sealed_16ms_truth_calls"] == 0,
    }
    arrays = {
        "checkpoint_coordinate_jacobian470x560": coordinate_jacobian,
        "checkpoint_gauge_basis560x90": gauge_basis,
        "checkpoint_augmented_chart_jacobian560x560": augmented,
        "initial_hidden_basis388x8": B0,
        "rank8_hidden_tangent_step_ladder": np.asarray(step_hidden_responses),
        "rank8_coordinate_tangent_step_ladder": np.asarray(
            step_coordinate_responses
        ),
        "rank8_coordinate_hessian_action_step_ladder": np.asarray(
            step_hessian_actions
        ),
        "rank8_physical_direction_scalings_step_ladder": np.asarray(
            direction_scales
        ),
        "selected_hidden_basis388": selected_basis,
        "selected_hidden_tangent_action388": selected_hidden_response,
        "selected_reduced_tangent_per_second": reduced_tangent,
        "hidden_basis_Z388": Z,
        "hidden_dual_Q388": Q,
        "macro_restriction_R82": R,
    }
    return metrics, arrays, checks


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "DIAGNOSTIC_PASS" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": status,
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
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("transition hidden-tangent result already exists")
    metrics, arrays, checks = _execute()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "transition_hidden_tangent_arrays.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "transition_hidden_tangent_metrics.json",
        {"metrics": metrics, "checks": checks, "passed": metrics["passed"]},
    )
    _write_json(CANONICAL_DIRECTORY / "input_execution_contract.json", _read(PARENT_CONTRACT))
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            **frozen,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "selected_hidden_rank": metrics["selected_rank_metrics"]["rank"],
        "hidden_tangent_invariance_relative_defect": metrics[
            "selected_rank_metrics"
        ]["hidden_tangent_invariance_relative_defect"],
        "hidden_physical_tangent_energy_capture": metrics[
            "selected_rank_metrics"
        ]["hidden_physical_tangent_energy_capture"],
        "maximum_rank8_hidden_tangent_step_relative_defect": metrics[
            "maximum_rank8_hidden_tangent_step_relative_defect"
        ],
        "coordinate_Jacobian_evaluations": metrics[
            "coordinate_Jacobian_evaluations"
        ],
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "new_chart_retractions": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "physical_fixed_Q_failure_detected": False,
        "full470_offline_transition_reference_preserved": True,
        "transition_trajectory_authorized": False,
        "online_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": (
                "DIAGNOSTIC_PASS" if summary["passed"] else "REJECTED"
            ),
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative)
                for relative in (
                    THIS_RUNNER,
                    THIS_TEST,
                    manifest.THIS_RUNNER,
                    manifest.THIS_TEST,
                    manifest.EXACT_CHART_RUNNER,
                    manifest.FIXED_Q_SOURCE,
                    manifest.TANGENT_SOURCE,
                )
            },
            "decisive_input_hashes": {
                "parent_contract": _sha(PARENT_CONTRACT),
                "parent_basis_arrays": _sha(PARENT_BASIS_ARRAYS),
                "saved_generator": _sha(manifest.SAVED_GENERATOR),
                "saved_checkpoint": _sha(manifest.SAVED_CHECKPOINT),
                "field_arrays": _sha(FIELD_ARRAYS),
                "current_chart_arrays": _sha(CURRENT_CHART_ARRAYS),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    selected = metrics["selected_rank_metrics"]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Transition hidden tangent WP10c9d6c7c3b5c4f25dk",
                "",
                "## Classification",
                "",
                f"`{metrics['classification']}`",
                "",
                f"The saved-generator diagnostic selected rank `{selected['rank']}`. Its hidden tangent invariance defect is `{selected['hidden_tangent_invariance_relative_defect']:.6e}` and its gauge-fixed physical tangent energy capture is `{selected['hidden_physical_tangent_energy_capture']:.12%}`.",
                "",
                f"The three-step coordinate-Hessian audit has maximum adjacent hidden-response defect `{metrics['maximum_rank8_hidden_tangent_step_relative_defect']:.6e}`. The execution used `{metrics['coordinate_Jacobian_evaluations']}` coordinate-Jacobian evaluations and zero new fixed-Q rates, complete generators, roots, chart retractions, propagated states, or sealed 16 ms truth calls.",
                "",
                "The reduced spectrum is diagnostic only because this is an unclassified transition checkpoint, not a branch equilibrium. The full y470 transition reference remains mandatory.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`. No transition trajectory, truth campaign, branch root, online transition ODE, microburst, or reduced slow evolution is authorized by this diagnostic.",
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
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
